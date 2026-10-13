from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg, in_bounds, neighbors4


BG = 0
MARKER_COLORS = (1, 2, 3)
CONNECTOR = 5
SPRITE = 8
SIDES = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
SIDE_STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

GEN_ATTEMPTS = 400


def _normalize_shape(coords: set[tuple[int, int]], connector: tuple[int, int]):
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, (connector[0] - r0, connector[1] - c0)


def _shape_bbox(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    return bbox_coords(np.array(coords, dtype=int))


def _connector_side(connector: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = map(int, connector)
    r0, c0, r1, c1 = map(int, bbox)
    sides = []
    if r == r0:
        sides.append("up")
    if r == r1:
        sides.append("down")
    if c == c0:
        sides.append("left")
    if c == c1:
        sides.append("right")
    assert len(sides) == 1, "connector must lie on exactly one bbox edge"
    return sides[0]


def _translate_coords(coords: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r + dr), int(c + dc)) for r, c in coords))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _bbox_shift(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _make_shape(side: str) -> dict:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        max_area = min(12, h * w - 1)
        if max_area < 6:
            continue
        area = random.randint(6, max_area)

        if side == "up":
            connector = (0, random.randint(1, w - 2))
            inward = (1, connector[1])
        elif side == "down":
            connector = (h - 1, random.randint(1, w - 2))
            inward = (h - 2, connector[1])
        elif side == "left":
            connector = (random.randint(1, h - 2), 0)
            inward = (connector[0], 1)
        else:
            connector = (random.randint(1, h - 2), w - 1)
            inward = (connector[0], w - 2)

        cells = {connector, inward}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        norm_coords, norm_connector = _normalize_shape(cells, connector)
        bbox = _shape_bbox(norm_coords)
        if len(norm_coords) == (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1):
            continue
        try:
            norm_side = _connector_side(norm_connector, bbox)
        except AssertionError:
            continue
        if norm_side != side:
            continue
        return {
            "coords": norm_coords,
            "connector": norm_connector,
            "bbox": bbox,
            "side": side,
        }
    raise RuntimeError("failed to generate a valid sprite shape")


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    allowed = {BG, SPRITE, CONNECTOR, *MARKER_COLORS}
    assert set(map(int, np.unique(g))) <= allowed

    sprite_components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        colors=[SPRITE, CONNECTOR],
        return_format="dict",
    )
    assert len(sprite_components) in (4, 6)

    sprite_components = sorted(sprite_components, key=lambda comp: tuple(map(int, comp["bbox"][:2])))
    coord_to_sprite: dict[tuple[int, int], int] = {}
    sprites = []
    for idx, comp in enumerate(sprite_components):
        coords = tuple(sorted((int(r), int(c)) for r, c in comp["coords"]))
        connector_cells = [coord for coord in coords if int(g[coord]) == CONNECTOR]
        assert len(connector_cells) == 1
        bbox = tuple(map(int, comp["bbox"]))
        side = _connector_side(connector_cells[0], bbox)
        for coord in coords:
            assert coord not in coord_to_sprite
            coord_to_sprite[coord] = idx
        sprites.append(
            {
                "id": idx,
                "coords": coords,
                "bbox": bbox,
                "connector": connector_cells[0],
                "connector_side": side,
            }
        )

    sprite_marker: dict[int, tuple[int, int, int]] = {}
    for r, c in np.argwhere(np.isin(g, list(MARKER_COLORS))):
        rr, cc = int(r), int(c)
        touched = {coord_to_sprite[(nr, nc)] for nr, nc in neighbors4(rr, cc) if (nr, nc) in coord_to_sprite}
        assert len(touched) == 1
        sprite_id = next(iter(touched))
        assert sprite_id not in sprite_marker
        sprite_marker[sprite_id] = (rr, cc, int(g[rr, cc]))
    assert len(sprite_marker) == len(sprites)

    color_to_ids: dict[int, list[int]] = {}
    for sprite in sprites:
        marker_r, marker_c, marker_color = sprite_marker[sprite["id"]]
        sprite["marker"] = (marker_r, marker_c)
        sprite["marker_color"] = marker_color
        color_to_ids.setdefault(marker_color, []).append(sprite["id"])

    assert len(color_to_ids) * 2 == len(sprites)
    pairs = []
    for color, ids in sorted(color_to_ids.items()):
        assert len(ids) == 2
        ids = sorted(ids, key=lambda idx: tuple(map(int, sprites[idx]["bbox"][:2])))
        base_id, insert_id = ids
        base = sprites[base_id]
        insert = sprites[insert_id]
        assert insert["connector_side"] == OPPOSITE[base["connector_side"]]
        pairs.append({"color": int(color), "base": base_id, "insert": insert_id})

    return sprites, pairs, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair sprites by transferred marker color, then dock each pair according to the base connector edge.

    Input structure:
    - The grid is mostly black background and contains 4 or 6 separated teal 4-connected sprites.
    - Each sprite contains exactly one gray connector pixel on exactly one bbox edge and exactly one adjacent one-cell marker in red, blue, or green.
    - Each marker touches only one sprite, and each marker color appears on exactly two sprites.

    Rule:
    - Read each sprite's transferred color from its unique 4-neighbor marker.
    - Pair the two sprites that received the same marker color.
    - In each pair, the sprite with lexicographically smaller bbox top-left stays fixed as the base.
    - Move the other sprite so its gray connector becomes 4-neighbor adjacent in the direction indicated by the base connector edge, then repaint both gray connectors teal and erase all markers.

    Key insight:
    - The gray connector says how to dock, but the external marker color says who docks with whom; both cues are required.
    """
    g = np.asarray(grid, dtype=int)
    sprites, pairs, bg = _parse_scene(g)
    out = np.full_like(g, bg)
    insert_ids = {pair["insert"] for pair in pairs}

    for sprite in sprites:
        if sprite["id"] in insert_ids:
            continue
        for r, c in sprite["coords"]:
            out[r, c] = SPRITE

    for pair in pairs:
        base = sprites[pair["base"]]
        insert = sprites[pair["insert"]]
        dr_step, dc_step = SIDE_STEP[base["connector_side"]]
        dr = int(base["connector"][0] + dr_step - insert["connector"][0])
        dc = int(base["connector"][1] + dc_step - insert["connector"][1])
        moved = _translate_coords(insert["coords"], dr, dc)
        for r, c in moved:
            assert in_bounds(out.shape, r, c)
            assert out[r, c] == bg
            out[r, c] = SPRITE

    return out


def _pair_geometry(base_shape: dict, insert_shape: dict) -> dict:
    step = SIDE_STEP[base_shape["side"]]
    out_shift = (
        base_shape["connector"][0] + step[0] - insert_shape["connector"][0],
        base_shape["connector"][1] + step[1] - insert_shape["connector"][1],
    )
    pair_cells = set(base_shape["coords"])
    pair_cells.update((r + out_shift[0], c + out_shift[1]) for r, c in insert_shape["coords"])
    pair_bbox = bbox_coords(np.array(sorted(pair_cells), dtype=int))
    return {"output_shift": out_shift, "pair_cells": tuple(sorted(pair_cells)), "pair_bbox": pair_bbox}


def _candidate_positions_for_bbox(grid_shape: tuple[int, int], bbox: tuple[int, int, int, int]):
    h, w = map(int, grid_shape)
    r0, c0, r1, c1 = map(int, bbox)
    return [
        (r, c)
        for r in range(-r0, h - r1)
        for c in range(-c0, w - c1)
    ]


def _global_bbox_for_coords(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    return bbox_coords(np.array(coords, dtype=int))


def _choose_insert_position(
    pair: dict,
    grid_shape: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    occupied_bboxes: list[tuple[int, int, int, int]],
    prefer: str | None = None,
    target: tuple[int, int] | None = None,
):
    h, w = map(int, grid_shape)
    local_bbox = pair["insert_shape"]["bbox"]
    base_top_left = pair["base_pos"]
    candidates = []
    for r in range(0, h - (local_bbox[2] - local_bbox[0])):
        for c in range(0, w - (local_bbox[3] - local_bbox[1])):
            if (r, c) <= base_top_left:
                continue
            placed = _translate_coords(pair["insert_shape"]["coords"], r, c)
            bbox = _global_bbox_for_coords(placed)
            if any(coord in occupied_cells for coord in placed):
                continue
            if any(not _bbox_separated(bbox, other_bbox, 3) for other_bbox in occupied_bboxes):
                continue
            score = 0
            if target is not None:
                placed_connector = (
                    r + pair["insert_shape"]["connector"][0],
                    c + pair["insert_shape"]["connector"][1],
                )
                score = _manhattan(placed_connector, target)
            candidates.append((score, (r, c), placed, bbox))
    if not candidates:
        return None
    if prefer == "near":
        candidates.sort(key=lambda item: (item[0], item[1]))
        pool = candidates[: max(1, len(candidates) // 5)]
    elif prefer == "far":
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        pool = candidates[: max(1, len(candidates) // 5)]
    else:
        pool = candidates
    return random.choice(pool)


def _marker_candidates(
    sprite_id: int,
    cell_to_sprite: dict[tuple[int, int], int],
    shape: tuple[int, int],
    used: set[tuple[int, int]],
):
    h, w = map(int, shape)
    candidates = set()
    for r, c in [coord for coord, owner in cell_to_sprite.items() if owner == sprite_id]:
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc):
                continue
            if (nr, nc) in cell_to_sprite or (nr, nc) in used:
                continue
            touched = {cell_to_sprite[(ar, ac)] for ar, ac in neighbors4(nr, nc) if (ar, ac) in cell_to_sprite}
            if touched == {sprite_id}:
                candidates.add((nr, nc))
    return sorted(candidates)


def _render_input(scene: dict) -> np.ndarray:
    grid = np.full(scene["shape"], BG, dtype=int)
    for sprite in scene["sprites"]:
        connector = sprite["connector"]
        for coord in sprite["coords"]:
            grid[coord] = CONNECTOR if coord == connector else SPRITE
        mr, mc = sprite["marker"]
        grid[mr, mc] = sprite["marker_color"]
    return grid


def _render_expected_output(scene: dict) -> np.ndarray:
    grid = np.full(scene["shape"], BG, dtype=int)
    for pair in scene["pairs"]:
        base = scene["sprites"][pair["base"]]
        for coord in base["coords"]:
            grid[coord] = SPRITE
    for pair in scene["pairs"]:
        insert = scene["sprites"][pair["insert"]]
        for coord in pair["output_coords"]:
            assert grid[coord] == BG
            grid[coord] = SPRITE
    return grid


def _scene_is_non_degenerate(scene: dict) -> bool:
    sprites = scene["sprites"]
    pairs = scene["pairs"]

    side_counts: dict[str, int] = {}
    for pair in pairs:
        side_counts[pair["base_side"]] = side_counts.get(pair["base_side"], 0) + 1
    if max(side_counts.values()) < 2:
        return False

    if not any(pair["base_shape_key"] != pair["insert_shape_key"] for pair in pairs):
        return False

    compatible_decoy = False
    for pair in pairs:
        required_side = OPPOSITE[pair["base_side"]]
        candidate_count = sum(
            1
            for sprite in sprites
            if sprite["id"] != pair["base"] and sprite["connector_side"] == required_side
        )
        if candidate_count >= 2:
            compatible_decoy = True
            break
    if not compatible_decoy:
        return False

    nearest_mismatch = False
    for sprite in sprites:
        distances = []
        for other in sprites:
            if other["id"] == sprite["id"]:
                continue
            distances.append(
                (
                    _manhattan(sprite["connector"], other["connector"]),
                    tuple(map(int, other["bbox"][:2])),
                    other["id"],
                )
            )
        distances.sort()
        if not distances:
            return False
        nearest_mismatch |= distances[0][2] != sprite["partner"]
    return nearest_mismatch


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 2, 3))
    side = random.choice(SIDES)
    base_sides = [side, side]
    if pair_count == 3:
        base_sides.append(random.choice(SIDES))
    random.shuffle(base_sides)

    colors = random.sample(list(MARKER_COLORS), pair_count)
    pair_defs = []
    for pair_index in range(pair_count):
        base_shape = _make_shape(base_sides[pair_index])
        insert_shape = _make_shape(OPPOSITE[base_sides[pair_index]])
        geometry = _pair_geometry(base_shape, insert_shape)
        pair_defs.append(
            {
                "pair_index": pair_index,
                "marker_color": colors[pair_index],
                "base_shape": base_shape,
                "insert_shape": insert_shape,
                "base_side": base_shape["side"],
                "base_shape_key": tuple(base_shape["coords"]),
                "insert_shape_key": tuple(insert_shape["coords"]),
                **geometry,
            }
        )

    if pair_count == 2:
        shape = (random.randint(20, 26), random.randint(20, 26))
    else:
        shape = (random.randint(22, 29), random.randint(22, 29))

    occupied_output_cells: set[tuple[int, int]] = set()
    occupied_output_bboxes: list[tuple[int, int, int, int]] = []
    occupied_input_bboxes: list[tuple[int, int, int, int]] = []
    base_sprite_cells: set[tuple[int, int]] = set()

    for pair in sorted(pair_defs, key=lambda item: len(item["pair_cells"]), reverse=True):
        candidates = _candidate_positions_for_bbox(shape, pair["pair_bbox"])
        random.shuffle(candidates)
        placed = False
        for base_r, base_c in candidates:
            output_cells = {(r + base_r, c + base_c) for r, c in pair["pair_cells"]}
            output_bbox = _bbox_shift(pair["pair_bbox"], base_r, base_c)
            if any(coord in occupied_output_cells for coord in output_cells):
                continue
            if any(not _bbox_separated(output_bbox, other_bbox, 1) for other_bbox in occupied_output_bboxes):
                continue

            base_coords = _translate_coords(pair["base_shape"]["coords"], base_r, base_c)
            base_bbox = _global_bbox_for_coords(base_coords)
            if any(not _bbox_separated(base_bbox, other_bbox, 3) for other_bbox in occupied_input_bboxes):
                continue

            pair["base_pos"] = (base_r, base_c)
            pair["base_coords"] = base_coords
            pair["base_bbox"] = base_bbox
            pair["base_connector"] = (
                base_r + pair["base_shape"]["connector"][0],
                base_c + pair["base_shape"]["connector"][1],
            )
            pair["output_bbox"] = output_bbox
            pair["output_coords"] = tuple(sorted((r + base_r, c + base_c) for r, c in pair["pair_cells"] if (r, c) not in pair["base_shape"]["coords"]))
            occupied_output_cells |= output_cells
            occupied_output_bboxes.append(output_bbox)
            occupied_input_bboxes.append(base_bbox)
            base_sprite_cells |= set(base_coords)
            placed = True
            break
        if not placed:
            return None

    pair_by_index = {pair["pair_index"]: pair for pair in pair_defs}
    occupied_input_cells = set(base_sprite_cells)

    for pair in pair_defs:
        pair["insert_pos"] = None
        pair["insert_coords"] = None
        pair["insert_bbox"] = None
        pair["insert_connector"] = None

    if pair_count >= 2:
        anchor_pair = pair_by_index[0]
        decoy_pair = pair_by_index[1]
        near_pick = _choose_insert_position(
            decoy_pair,
            shape,
            occupied_input_cells,
            occupied_input_bboxes,
            prefer="near",
            target=anchor_pair["base_connector"],
        )
        if near_pick is not None:
            _score, pos, coords, bbox = near_pick
            decoy_pair["insert_pos"] = pos
            decoy_pair["insert_coords"] = coords
            decoy_pair["insert_bbox"] = bbox
            decoy_pair["insert_connector"] = (
                pos[0] + decoy_pair["insert_shape"]["connector"][0],
                pos[1] + decoy_pair["insert_shape"]["connector"][1],
            )
            occupied_input_cells |= set(coords)
            occupied_input_bboxes.append(bbox)

        far_pick = _choose_insert_position(
            anchor_pair,
            shape,
            occupied_input_cells,
            occupied_input_bboxes,
            prefer="far",
            target=anchor_pair["base_connector"],
        )
        if far_pick is not None:
            _score, pos, coords, bbox = far_pick
            anchor_pair["insert_pos"] = pos
            anchor_pair["insert_coords"] = coords
            anchor_pair["insert_bbox"] = bbox
            anchor_pair["insert_connector"] = (
                pos[0] + anchor_pair["insert_shape"]["connector"][0],
                pos[1] + anchor_pair["insert_shape"]["connector"][1],
            )
            occupied_input_cells |= set(coords)
            occupied_input_bboxes.append(bbox)

    for pair in pair_defs:
        if pair["insert_coords"] is not None:
            continue
        pick = _choose_insert_position(
            pair,
            shape,
            occupied_input_cells,
            occupied_input_bboxes,
            prefer=random.choice((None, "near", "far")),
            target=pair["base_connector"],
        )
        if pick is None:
            return None
        _score, pos, coords, bbox = pick
        pair["insert_pos"] = pos
        pair["insert_coords"] = coords
        pair["insert_bbox"] = bbox
        pair["insert_connector"] = (
            pos[0] + pair["insert_shape"]["connector"][0],
            pos[1] + pair["insert_shape"]["connector"][1],
        )
        occupied_input_cells |= set(coords)
        occupied_input_bboxes.append(bbox)

    sprites = []
    sprite_id = 0
    for pair in pair_defs:
        base_sprite = {
            "id": sprite_id,
            "pair_index": pair["pair_index"],
            "role": "base",
            "coords": pair["base_coords"],
            "bbox": pair["base_bbox"],
            "connector": pair["base_connector"],
            "connector_side": pair["base_side"],
            "marker_color": pair["marker_color"],
            "shape_key": pair["base_shape_key"],
        }
        sprite_id += 1
        insert_sprite = {
            "id": sprite_id,
            "pair_index": pair["pair_index"],
            "role": "insert",
            "coords": pair["insert_coords"],
            "bbox": pair["insert_bbox"],
            "connector": pair["insert_connector"],
            "connector_side": OPPOSITE[pair["base_side"]],
            "marker_color": pair["marker_color"],
            "shape_key": pair["insert_shape_key"],
        }
        sprite_id += 1
        sprites.extend([base_sprite, insert_sprite])
        pair["base"] = base_sprite["id"]
        pair["insert"] = insert_sprite["id"]

    by_pair = {}
    for sprite in sprites:
        by_pair.setdefault(sprite["pair_index"], []).append(sprite)
    for members in by_pair.values():
        assert len(members) == 2
        a, b = members
        a["partner"] = b["id"]
        b["partner"] = a["id"]

    cell_to_sprite = {}
    for sprite in sprites:
        for coord in sprite["coords"]:
            if coord in cell_to_sprite:
                return None
            cell_to_sprite[coord] = sprite["id"]

    used_markers: set[tuple[int, int]] = set()
    for sprite in sprites:
        candidates = _marker_candidates(sprite["id"], cell_to_sprite, shape, used_markers)
        if not candidates:
            return None
        sprite["marker"] = random.choice(candidates)
        used_markers.add(sprite["marker"])

    scene = {"shape": shape, "sprites": sprites, "pairs": pair_defs}
    expected = _render_expected_output(scene)
    inp = _render_input(scene)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _scene_is_non_degenerate(scene):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
