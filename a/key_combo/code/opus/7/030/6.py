from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
CONTROL = 5
MAIN_COLORS = (1, 2, 3, 4)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intersect_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _paint(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    patch = canvas[top : top + h, left : left + w]
    mask = sprite != BG
    patch[mask] = sprite[mask]


def _infer_sprites(grid: np.ndarray) -> list[dict]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, CONTROL))
    assert 3 <= len(colors) <= 4, "expected 3 or 4 distinct sprite colors"

    sprites: list[dict] = []
    for color in colors:
        bbox = _bbox_from_mask(grid == color)
        r0, c0, r1, c1 = bbox

        left_rows = [r - r0 for r in range(r0, r1 + 1) if int(grid[r, c0]) == CONTROL]
        right_rows = [r - r0 for r in range(r0, r1 + 1) if int(grid[r, c1]) == CONTROL]
        assert left_rows, f"sprite {color} must expose a left-edge grey marker"
        assert right_rows, f"sprite {color} must expose a right-edge grey marker"

        sprites.append(
            {
                "color": color,
                "bbox": bbox,
                "h": _bbox_h(bbox),
                "w": _bbox_w(bbox),
                "left_rows": left_rows,
                "right_rows": right_rows,
            }
        )
    return sprites


def _layer_order(grid: np.ndarray, sprites: list[dict]) -> list[dict]:
    colors = [sprite["color"] for sprite in sprites]
    by_color = {sprite["color"]: sprite for sprite in sprites}
    overlap_neighbors: dict[int, set[int]] = {color: set() for color in colors}
    below_edges: set[tuple[int, int]] = set()

    for color_a, color_b in itertools.combinations(colors, 2):
        bbox_a = by_color[color_a]["bbox"]
        bbox_b = by_color[color_b]["bbox"]
        inter = _intersect_bbox(bbox_a, bbox_b)
        if inter is None:
            continue

        overlap_neighbors[color_a].add(color_b)
        overlap_neighbors[color_b].add(color_a)
        r0, c0, r1, c1 = inter
        visible = {
            int(v)
            for v in np.unique(grid[r0 : r1 + 1, c0 : c1 + 1])
            if int(v) in (color_a, color_b)
        }
        assert len(visible) <= 1, "a pairwise overlap must reveal at most one visible layer"
        if not visible:
            continue

        higher = next(iter(visible))
        lower = color_b if higher == color_a else color_a
        below_edges.add((lower, higher))

    valid_orders: list[tuple[int, ...]] = []
    for order in itertools.permutations(colors):
        positions = {color: i for i, color in enumerate(order)}
        if any(positions[lower] >= positions[higher] for lower, higher in below_edges):
            continue
        if any(order[i + 1] not in overlap_neighbors[order[i]] for i in range(len(order) - 1)):
            continue
        valid_orders.append(order)

    assert len(valid_orders) == 1, "layer order must be uniquely determined"
    return [by_color[color] for color in valid_orders[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden bottom-to-top order of overlapping grey-edged sprites and pack them left-to-right.

    Input structure
    - A black background contains 3 or 4 filled rectangular sprites with distinct main colors plus grey edge markers.
    - Each sprite exposes at least one grey pixel on its left edge and at least one grey pixel on its right edge.
    - Sprite bounding boxes partially overlap in the input, and the visible color in each overlap comes from the higher layer.

    Rule
    - Recover each sprite as its full bounding rectangle, treating any covered cells inside that bbox as belonging to the sprite's main color.
    - Infer the unique bottom-to-top layer order from the visible colors inside pairwise overlaps, with consecutive layers required to overlap.
    - Recolor every grey marker to its sprite's main color, then concatenate the cropped sprites from bottom layer to top layer.
    - Keep the bottom sprite flush with the output top; at each seam align the previous sprite's lowest right-edge grey row with the next sprite's highest left-edge grey row.

    Key insight
    - The input x-positions are a distraction: overlap colors determine the sprite order, while the grey edge markers determine only the vertical seam offsets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ordered = _layer_order(g, _infer_sprites(g))
    tops = [0]
    for prev, curr in zip(ordered, ordered[1:]):
        next_top = tops[-1] + max(prev["right_rows"]) - min(curr["left_rows"])
        assert next_top >= 0, "output is anchored to the first sprite's top row"
        tops.append(int(next_top))

    out_h = max(top + int(sprite["h"]) for top, sprite in zip(tops, ordered))
    out_w = sum(int(sprite["w"]) for sprite in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)

    left = 0
    for top, sprite in zip(tops, ordered):
        out[top : top + sprite["h"], left : left + sprite["w"]] = int(sprite["color"])
        left += int(sprite["w"])
    return out


GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 80
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (16, 24)
SPRITE_H_RANGE = (4, 6)
SPRITE_W_RANGE = (3, 5)
EXTRA_MARKER_PROB = 0.35


def _sample_edge_rows(height: int) -> list[int]:
    rows = {random.randrange(height)}
    if height >= 5 and random.random() < EXTRA_MARKER_PROB:
        rows.add(random.randrange(height))
    return sorted(rows)


def _make_sprite(spec: dict) -> np.ndarray:
    sprite = np.full((int(spec["h"]), int(spec["w"])), int(spec["color"]), dtype=int)
    sprite[np.array(spec["left_rows"], dtype=int), 0] = CONTROL
    sprite[np.array(spec["right_rows"], dtype=int), sprite.shape[1] - 1] = CONTROL
    return sprite


def _output_tops(specs: list[dict]) -> list[int] | None:
    tops = [0]
    for prev, curr in zip(specs, specs[1:]):
        top = tops[-1] + max(prev["right_rows"]) - min(curr["left_rows"])
        if top < 0:
            return None
        tops.append(int(top))
    return tops


def _render_expected_output(specs: list[dict]) -> np.ndarray | None:
    tops = _output_tops(specs)
    if tops is None:
        return None
    out_h = max(top + int(spec["h"]) for top, spec in zip(tops, specs))
    out_w = sum(int(spec["w"]) for spec in specs)
    if out_h > 30 or out_w > 30:
        return None
    out = np.full((out_h, out_w), BG, dtype=int)
    left = 0
    for top, spec in zip(tops, specs):
        out[top : top + spec["h"], left : left + spec["w"]] = int(spec["color"])
        left += int(spec["w"])
    return out


def _sample_specs() -> list[dict]:
    count = random.choice((3, 4))
    colors = random.sample(MAIN_COLORS, count)
    specs: list[dict] = []
    for color in colors:
        height = random.randint(*SPRITE_H_RANGE)
        width = random.randint(*SPRITE_W_RANGE)
        specs.append(
            {
                "color": int(color),
                "h": int(height),
                "w": int(width),
                "left_rows": _sample_edge_rows(height),
                "right_rows": _sample_edge_rows(height),
            }
        )
    return specs


def _sample_chain_positions(specs: list[dict]) -> list[tuple[int, int]] | None:
    positions = [(0, 0)]
    for prev, curr in zip(specs, specs[1:]):
        prev_r, prev_c = positions[-1]
        options: list[tuple[int, int]] = []
        for overlap_h in range(1, min(int(prev["h"]), int(curr["h"]))):
            for overlap_w in range(1, min(int(prev["w"]), int(curr["w"]))):
                for horiz in (-1, 1):
                    if horiz == 1:
                        prev_forbidden = set(int(v) for v in prev["right_rows"])
                        curr_forbidden = set(int(v) for v in curr["left_rows"])
                        curr_c = prev_c + int(prev["w"]) - overlap_w
                        prev_cols = range(int(prev["w"]) - overlap_w, int(prev["w"]))
                        curr_cols = range(0, overlap_w)
                    else:
                        prev_forbidden = set(int(v) for v in prev["left_rows"])
                        curr_forbidden = set(int(v) for v in curr["right_rows"])
                        curr_c = prev_c - int(curr["w"]) + overlap_w
                        prev_cols = range(0, overlap_w)
                        curr_cols = range(int(curr["w"]) - overlap_w, int(curr["w"]))

                    for vert in (-1, 1):
                        if vert == 1:
                            prev_rows = range(int(prev["h"]) - overlap_h, int(prev["h"]))
                            curr_rows = range(0, overlap_h)
                            curr_r = prev_r + int(prev["h"]) - overlap_h
                        else:
                            prev_rows = range(0, overlap_h)
                            curr_rows = range(int(curr["h"]) - overlap_h, int(curr["h"]))
                            curr_r = prev_r - int(curr["h"]) + overlap_h

                        if set(prev_rows) & prev_forbidden:
                            continue
                        if set(curr_rows) & curr_forbidden:
                            continue
                        options.append((curr_r, curr_c))
        if not options:
            return None
        positions.append(random.choice(options))
    return positions


def _fit_positions(
    specs: list[dict], positions: list[tuple[int, int]]
) -> tuple[int, int, list[tuple[int, int]]] | None:
    min_r = min(r for r, _c in positions)
    min_c = min(c for _r, c in positions)
    shifted = [(r - min_r, c - min_c) for r, c in positions]

    need_h = max(r + int(spec["h"]) for (r, _c), spec in zip(shifted, specs))
    need_w = max(c + int(spec["w"]) for (_r, c), spec in zip(shifted, specs))
    if need_h > GRID_H_RANGE[1] or need_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(max(GRID_H_RANGE[0], need_h), GRID_H_RANGE[1])
    grid_w = random.randint(max(GRID_W_RANGE[0], need_w), GRID_W_RANGE[1])
    pad_r = random.randint(0, grid_h - need_h)
    pad_c = random.randint(0, grid_w - need_w)
    fitted = [(r + pad_r, c + pad_c) for r, c in shifted]
    return grid_h, grid_w, fitted


def _latent_bboxes(
    specs: list[dict], positions: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for (top, left), spec in zip(positions, specs):
        bboxes.append((top, left, top + int(spec["h"]) - 1, left + int(spec["w"]) - 1))
    return bboxes


def _all_overlaps_are_interior(bboxes: list[tuple[int, int, int, int]]) -> bool:
    for bbox_a, bbox_b in itertools.combinations(bboxes, 2):
        inter = _intersect_bbox(bbox_a, bbox_b)
        if inter is None:
            continue
        if (
            inter[0] <= bbox_a[0]
            or inter[2] >= bbox_a[2]
            or inter[1] <= bbox_a[1]
            or inter[3] >= bbox_a[3]
            or inter[0] <= bbox_b[0]
            or inter[2] >= bbox_b[2]
            or inter[1] <= bbox_b[1]
            or inter[3] >= bbox_b[3]
        ):
            return False
    return True


def _non_adjacent_disjoint(bboxes: list[tuple[int, int, int, int]]) -> bool:
    for i in range(len(bboxes)):
        for j in range(i + 2, len(bboxes)):
            if _intersect_bbox(bboxes[i], bboxes[j]) is not None:
                return False
    return True


def _x_order_differs_from_layer_order(positions: list[tuple[int, int]]) -> bool:
    x_order = sorted(range(len(positions)), key=lambda i: (positions[i][1], positions[i][0]))
    return x_order != list(range(len(positions)))


def _render_input(specs: list[dict], grid_h: int, grid_w: int, positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec, (top, left) in zip(specs, positions):
        _paint(grid, _make_sprite(spec), top, left)
    return grid


def _markers_stay_visible(inp: np.ndarray, specs: list[dict], positions: list[tuple[int, int]]) -> bool:
    for spec, (top, left) in zip(specs, positions):
        right_col = left + int(spec["w"]) - 1
        for rr in spec["left_rows"]:
            if int(inp[top + rr, left]) != CONTROL:
                return False
        for rr in spec["right_rows"]:
            if int(inp[top + rr, right_col]) != CONTROL:
                return False
    return True


def _visible_bboxes_match(
    inp: np.ndarray, specs: list[dict], bboxes: list[tuple[int, int, int, int]]
) -> bool:
    for spec, bbox in zip(specs, bboxes):
        mask = inp == int(spec["color"])
        if not np.any(mask):
            return False
        if _bbox_from_mask(mask) != bbox:
            return False
    return True


def _adjacent_overlaps_reveal_higher(
    inp: np.ndarray, specs: list[dict], bboxes: list[tuple[int, int, int, int]]
) -> bool:
    for i in range(len(specs) - 1):
        inter = _intersect_bbox(bboxes[i], bboxes[i + 1])
        if inter is None:
            return False
        r0, c0, r1, c1 = inter
        if not np.any(inp[r0 : r1 + 1, c0 : c1 + 1] == int(specs[i + 1]["color"])):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    expected = _render_expected_output(specs)
    if expected is None:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        base_positions = _sample_chain_positions(specs)
        if base_positions is None:
            continue
        fitted = _fit_positions(specs, base_positions)
        if fitted is None:
            continue
        grid_h, grid_w, positions = fitted
        bboxes = _latent_bboxes(specs, positions)

        if not _x_order_differs_from_layer_order(positions):
            continue
        if not _non_adjacent_disjoint(bboxes):
            continue

        inp = _render_input(specs, grid_h, grid_w, positions)
        if not _markers_stay_visible(inp, specs, positions):
            continue
        if not _visible_bboxes_match(inp, specs, bboxes):
            continue
        if not _adjacent_overlaps_reveal_higher(inp, specs, bboxes):
            continue

        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
