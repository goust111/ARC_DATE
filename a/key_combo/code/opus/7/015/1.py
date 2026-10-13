from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import (  # type: ignore
    Color,
    bbox_coords,
    components_mask,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


BG = Color.BLACK
MARKER_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE]
BODY_COLORS = [Color.BLUE, Color.TEAL, Color.MAROON]
LINE_COLOR_BY_SUM = {
    0: Color.GREY,
    1: Color.BLUE,
    2: Color.GREEN,
}

SHAPE_LIBRARY = {
    0: [
        np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    ],
    1: [
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=int),
        np.array([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], dtype=int),
    ],
    2: [
        np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
    ],
}

GEN_ATTEMPTS = 400


def _line_color(hole_sum: int) -> int:
    if hole_sum >= 3:
        return Color.RED
    return LINE_COLOR_BY_SUM[int(hole_sum)]


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _make_sprite(hole_count: int, body_color: int, marker_color: int) -> dict:
    mask = _transform_mask(random.choice(SHAPE_LIBRARY[hole_count]))
    occupied = np.argwhere(mask == 1)
    marker_rel = tuple(map(int, occupied[random.randrange(len(occupied))]))

    patch = np.zeros(mask.shape, dtype=int)
    patch[mask == 1] = int(body_color)
    patch[marker_rel] = int(marker_color)

    h, w = patch.shape
    center_offset = ((h - 1) // 2, (w - 1) // 2)
    return {
        "patch": patch,
        "hole_count": int(hole_count),
        "body_color": int(body_color),
        "marker_color": int(marker_color),
        "marker_rel": marker_rel,
        "center_offset": center_offset,
        "up": center_offset[0],
        "down": h - 1 - center_offset[0],
        "left": center_offset[1],
        "right": w - 1 - center_offset[1],
    }


def _bbox_from_center(center: tuple[int, int], sprite: dict) -> tuple[int, int, int, int]:
    cr, cc = center
    r0 = int(cr - sprite["up"])
    c0 = int(cc - sprite["left"])
    h, w = sprite["patch"].shape
    return r0, c0, r0 + h - 1, c0 + w - 1


def _top_left_from_center(center: tuple[int, int], sprite: dict) -> tuple[int, int]:
    cr, cc = center
    return int(cr - sprite["up"]), int(cc - sprite["left"])


def _coords_from_patch(patch: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    r0, c0 = top_left
    coords = np.argwhere(patch != BG)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _hole_count(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    holes = 0
    for comp in components_mask(~mask, connectivity=4):
        if (
            np.all(comp[:, 0] > 0)
            and np.all(comp[:, 0] < mask.shape[0] - 1)
            and np.all(comp[:, 1] > 0)
            and np.all(comp[:, 1] < mask.shape[1] - 1)
        ):
            holes += 1
    return int(holes)


def _state_from_coords(coords: np.ndarray, sprite_id: int, hole_count: int | None = None) -> dict:
    b = bbox_coords(coords)
    return {
        "id": int(sprite_id),
        "coords": np.array(coords, dtype=int),
        "bbox": b,
        "center": ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2),
        "hole_count": _hole_count(coords) if hole_count is None else int(hole_count),
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=False)

    sprites: list[dict] = []
    controls: dict[int, tuple[int, int]] = {}
    for comp in comps:
        coords = np.array(comp["coords"], dtype=int)
        if len(coords) == 1:
            r, c = map(int, coords[0])
            color = int(g[r, c])
            assert color not in controls, "control colors must be unique"
            controls[color] = (r, c)
            continue

        counts = Counter(int(g[r, c]) for r, c in coords)
        singles = [color for color, count in counts.items() if count == 1]
        repeated = [color for color, count in counts.items() if count > 1]
        assert len(counts) == 2 and len(singles) == 1 and len(repeated) == 1
        marker_color = int(singles[0])
        body_color = int(repeated[0])
        marker_coord = next((int(r), int(c)) for r, c in coords if int(g[r, c]) == marker_color)
        sprite_id = len(sprites)
        sprites.append(
            {
                "id": sprite_id,
                "coords": coords,
                "bbox": comp["bbox"],
                "body_color": body_color,
                "marker_color": marker_color,
                "marker_coord": marker_coord,
            }
        )

    assert 4 <= len(sprites) <= 6
    assert len(controls) == len(sprites)
    for sprite in sprites:
        assert sprite["marker_color"] in controls
    return sprites, controls


def _apply_translation(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    sprites, controls = _parse_scene(g)

    base = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        base[coords[:, 0], coords[:, 1]] = BG
    for control in controls.values():
        base[control] = BG

    translated: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for sprite in sprites:
        target = controls[sprite["marker_color"]]
        mr, mc = sprite["marker_coord"]
        dr = int(target[0] - mr)
        dc = int(target[1] - mc)

        new_coords = []
        pixel_payload = []
        for r, c in sprite["coords"]:
            rr = int(r + dr)
            cc = int(c + dc)
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            new_coords.append((rr, cc))
            pixel_payload.append((rr, cc, int(g[r, c])))
            assert (rr, cc) not in occupied
            occupied.add((rr, cc))

        for rr, cc, color in pixel_payload:
            base[rr, cc] = int(color)

        moved = _state_from_coords(np.array(new_coords, dtype=int), sprite["id"])
        moved["marker_color"] = sprite["marker_color"]
        moved["body_color"] = sprite["body_color"]
        moved["marker_coord"] = target
        translated.append(moved)

    return base, translated, sprites, controls


def _compute_connectors(sprites: list[dict], base_grid: np.ndarray) -> list[dict]:
    connectors: list[dict] = []

    row_groups: dict[int, list[dict]] = {}
    for sprite in sprites:
        row_groups.setdefault(int(sprite["center"][0]), []).append(sprite)

    for row in sorted(row_groups):
        group = sorted(row_groups[row], key=lambda item: int(item["center"][1]))
        for left, right in zip(group, group[1:]):
            c_start = int(left["bbox"][3] + 1)
            c_stop = int(right["bbox"][1] - 1)
            if c_start > c_stop:
                continue
            cells = [(int(row), c) for c in range(c_start, c_stop + 1)]
            if all(int(base_grid[r, c]) == BG for r, c in cells):
                connectors.append(
                    {
                        "orientation": "h",
                        "cells": cells,
                        "pair": (int(left["id"]), int(right["id"])),
                        "color": _line_color(int(left["hole_count"]) + int(right["hole_count"])),
                    }
                )

    col_groups: dict[int, list[dict]] = {}
    for sprite in sprites:
        col_groups.setdefault(int(sprite["center"][1]), []).append(sprite)

    for col in sorted(col_groups):
        group = sorted(col_groups[col], key=lambda item: int(item["center"][0]))
        for top, bottom in zip(group, group[1:]):
            r_start = int(top["bbox"][2] + 1)
            r_stop = int(bottom["bbox"][0] - 1)
            if r_start > r_stop:
                continue
            cells = [(r, int(col)) for r in range(r_start, r_stop + 1)]
            if all(int(base_grid[r, c]) == BG for r, c in cells):
                connectors.append(
                    {
                        "orientation": "v",
                        "cells": cells,
                        "pair": (int(top["id"]), int(bottom["id"])),
                        "color": _line_color(int(top["hole_count"]) + int(bottom["hole_count"])),
                    }
                )

    return connectors


def _render_solution(base_grid: np.ndarray, connectors: list[dict]) -> np.ndarray:
    out = np.array(base_grid, copy=True)
    for connector in connectors:
        for r, c in connector["cells"]:
            out[r, c] = int(connector["color"])
    return out


def _solve_metadata(grid: np.ndarray) -> dict:
    base, translated, original_sprites, controls = _apply_translation(grid)
    connectors = _compute_connectors(translated, base)
    return {
        "base": base,
        "sprites": translated,
        "connectors": connectors,
        "output": _render_solution(base, connectors),
        "original_sprites": original_sprites,
        "controls": controls,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each marked sprite to its matching control, then connect aligned centers with hole-sum-colored lines.

    Input structure:
    - The grid has a black background, 4-6 separated 4-connected sprites, and one isolated control pixel for each sprite marker color.
    - Each sprite is monochrome except for one embedded marker pixel of a distinct color; the isolated controls reuse those marker colors one-to-one.
    - Some sprites are solid while others enclose one or more black holes inside their own bounding boxes.

    Rule:
    - Move each whole sprite so that its embedded marker lands exactly on the isolated control pixel of the same color, then erase the consumed controls and original sprite positions.
    - For every translated sprite, count its enclosed 4-connected black holes and compute the floor-centered coordinate of its bounding box.
    - Among sprites sharing a center row, connect only adjacent pairs in left-to-right order if the horizontal segment between their bounding boxes is entirely black; do the analogous top-to-bottom test for shared center columns.
    - Color each connector by the sum of the two endpoint hole counts: 0 -> grey, 1 -> blue, 2 -> green, 3 or more -> red; draw horizontal connectors before vertical ones so later vertical crossings overwrite.

    Key insight:
    - The marker-control pairing determines the spatial rearrangement, while the hole count is a separate topological invariant that only reappears when deciding connector colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG
    return _solve_metadata(g)["output"]


def _connector_signature(connectors: list[dict]) -> tuple:
    return tuple(
        sorted(
            (
                connector["orientation"],
                connector["pair"],
                connector["color"],
                tuple(connector["cells"]),
            )
            for connector in connectors
        )
    )


def _group_sizes(sprites: list[dict]) -> tuple[int, int]:
    row_counts: Counter[int] = Counter(int(sprite["center"][0]) for sprite in sprites)
    col_counts: Counter[int] = Counter(int(sprite["center"][1]) for sprite in sprites)
    return max(row_counts.values()), max(col_counts.values())


def _build_relative_centers(sprites: list[dict], scenario: str) -> dict[int, tuple[int, int]]:
    gap = lambda: random.randint(2, 5)
    centers: dict[int, tuple[int, int]] = {}

    if scenario == "center_cross_5":
        center, left, right, up, down = sprites
        centers[0] = (0, 0)
        centers[1] = (0, -(left["right"] + 1 + gap() + center["left"]))
        centers[2] = (0, center["right"] + 1 + gap() + right["left"])
        centers[3] = (-(up["down"] + 1 + gap() + center["up"]), 0)
        centers[4] = (center["down"] + 1 + gap() + down["up"], 0)
        return centers

    if scenario == "center_cross_6":
        center, left, right, up, down, extra = sprites
        centers = _build_relative_centers(sprites[:5], "center_cross_5")
        mode = random.choice(["row_left", "row_right", "col_up", "col_down"])
        if mode == "row_left":
            left_c = centers[1][1]
            centers[5] = (0, left_c - (extra["right"] + 1 + gap() + left["left"]))
        elif mode == "row_right":
            right_c = centers[2][1]
            centers[5] = (0, right_c + right["right"] + 1 + gap() + extra["left"])
        elif mode == "col_up":
            up_r = centers[3][0]
            centers[5] = (up_r - (extra["down"] + 1 + gap() + up["up"]), 0)
        else:
            down_r = centers[4][0]
            centers[5] = (down_r + down["down"] + 1 + gap() + extra["up"], 0)
        return centers

    if scenario == "empty_cross_5":
        far_left, left, right, top, bottom = sprites
        centers[1] = (0, -(left["right"] + 1 + random.randint(1, 3)))
        centers[2] = (0, right["left"] + 1 + random.randint(1, 3))
        centers[0] = (
            0,
            centers[1][1] - (far_left["right"] + 1 + gap() + left["left"]),
        )
        centers[3] = (-(top["down"] + 1 + gap()), 0)
        centers[4] = (bottom["up"] + 1 + gap(), 0)
        return centers

    if scenario == "empty_cross_6":
        far_left, left, right, top, bottom, far_right = sprites
        centers = _build_relative_centers(sprites[:5], "empty_cross_5")
        centers[5] = (
            0,
            centers[2][1] + right["right"] + 1 + gap() + far_right["left"],
        )
        return centers

    raise AssertionError(f"unknown scenario: {scenario}")


def _layout_final_positions(sprites: list[dict], scenario: str) -> tuple[tuple[int, int], list[dict]]:
    rel_centers = _build_relative_centers(sprites, scenario)
    rel_boxes = []
    for idx, sprite in enumerate(sprites):
        b = _bbox_from_center(rel_centers[idx], sprite)
        rel_boxes.append((idx, b))

    min_r = min(b[0] for _, b in rel_boxes)
    min_c = min(b[1] for _, b in rel_boxes)
    max_r = max(b[2] for _, b in rel_boxes)
    max_c = max(b[3] for _, b in rel_boxes)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    H = random.randint(max(18, span_h + 4), min(28, span_h + 10))
    W = random.randint(max(18, span_w + 4), min(28, span_w + 10))

    shift_r = random.randint(1 - min_r, H - 2 - max_r)
    shift_c = random.randint(1 - min_c, W - 2 - max_c)

    placed = []
    for idx, sprite in enumerate(sprites):
        center = (rel_centers[idx][0] + shift_r, rel_centers[idx][1] + shift_c)
        top_left = _top_left_from_center(center, sprite)
        marker_abs = (
            top_left[0] + sprite["marker_rel"][0],
            top_left[1] + sprite["marker_rel"][1],
        )
        placed.append(
            {
                **sprite,
                "id": idx,
                "center": center,
                "top_left": top_left,
                "bbox": _bbox_from_center(center, sprite),
                "marker_abs": marker_abs,
                "coords": _coords_from_patch(sprite["patch"], top_left),
            }
        )

    marker_positions = [item["marker_abs"] for item in placed]
    assert len(set(marker_positions)) == len(marker_positions)
    return (H, W), placed


def _place_initial_scene(shape: tuple[int, int], final_sprites: list[dict]) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    for sprite in final_sprites:
        mr, mc = sprite["marker_abs"]
        assert int(grid[mr, mc]) == BG
        grid[mr, mc] = int(sprite["marker_color"])

    shuffled = list(final_sprites)
    random.shuffle(shuffled)
    for sprite in shuffled:
        patch = sprite["patch"]
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=BG)

    return grid


def _original_scene_without_controls(grid: np.ndarray, controls: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for control in controls.values():
        out[control] = BG
    return out


def _validate_generated_input(grid: np.ndarray) -> None:
    meta = _solve_metadata(grid)
    connectors = meta["connectors"]
    assert len(connectors) >= 2
    assert len({int(conn["color"]) for conn in connectors}) >= 2
    assert any(int(sprite["hole_count"]) > 0 for sprite in meta["sprites"])

    row_max, col_max = _group_sizes(meta["sprites"])
    assert max(row_max, col_max) >= 3

    original_grid = _original_scene_without_controls(grid, meta["controls"])
    original_states = [
        _state_from_coords(np.array(sprite["coords"], dtype=int), int(sprite["id"]))
        for sprite in meta["original_sprites"]
    ]
    original_connectors = _compute_connectors(original_states, original_grid)
    assert _connector_signature(original_connectors) != _connector_signature(connectors)

    horizontal_cells = set()
    vertical_cells = set()
    for connector in connectors:
        if connector["orientation"] == "h":
            horizontal_cells.update(connector["cells"])
        else:
            vertical_cells.update(connector["cells"])
    if horizontal_cells & vertical_cells:
        assert any(conn["orientation"] == "v" for conn in connectors)


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([5, 6])
    marker_colors = random.sample(MARKER_COLORS, count)
    hole_counts = [random.choice([0, 1, 2]) for _ in range(count)]
    if max(hole_counts) == 0:
        hole_counts[random.randrange(count)] = random.choice([1, 2])

    body_colors = [random.choice(BODY_COLORS) for _ in range(count)]
    sprites = [
        _make_sprite(hole_counts[i], body_colors[i], marker_colors[i])
        for i in range(count)
    ]

    scenario = random.choice(
        ["center_cross_5", "empty_cross_5"] if count == 5 else ["center_cross_6", "empty_cross_6"]
    )

    try:
        shape, final_sprites = _layout_final_positions(sprites, scenario)
    except ValueError:
        return None

    base = np.zeros(shape, dtype=int)
    for sprite in final_sprites:
        try:
            paste(base, sprite["patch"], sprite["top_left"][0], sprite["top_left"][1], transparent=BG, overwrite=False)
        except ValueError:
            return None
    final_states = [
        _state_from_coords(np.array(sprite["coords"], dtype=int), int(sprite["id"]), int(sprite["hole_count"]))
        for sprite in final_sprites
    ]
    final_connectors = _compute_connectors(final_states, base)
    if len(final_connectors) < 2:
        return None
    if len({int(conn["color"]) for conn in final_connectors}) < 2:
        return None

    grid = _place_initial_scene(shape, final_sprites)
    if grid is None:
        return None

    try:
        _validate_generated_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
