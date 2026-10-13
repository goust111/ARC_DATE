from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, is_contiguous, paste


BG = Color.BLACK
PALETTE = tuple(Color.NOT_BLACK)

PAIR_COUNT_RANGE = (3, 4)
SHAPE_HEIGHT_RANGE = (2, 4)
SHAPE_WIDTH_RANGE = (3, 6)
PAIR_GAP_RANGE = (2, 6)
ROW_GAP_MIN = 2
GRID_MAX = 30
GEN_MAX_ATTEMPTS = 500


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = mask.astype(np.uint8)
    return tuple(map(int, arr.shape)), tuple(int(v) for v in arr.ravel())


def _family_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    base = _mask_key(mask)
    mirrored = _mask_key(np.fliplr(mask))
    return base if base <= mirrored else mirrored


def _mask_to_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _random_shape() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(*SHAPE_HEIGHT_RANGE)
        w = random.randint(*SHAPE_WIDTH_RANGE)
        area_lo = max(4, (h * w + 2) // 3)
        area_hi = min(h * w - 1, 8)
        if area_lo > area_hi:
            continue
        target_area = random.randint(area_lo, area_hi)

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        frontier = set(_neighbors4(*seed, h, w))
        while int(mask.sum()) < target_area:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for nb in _neighbors4(*cell, h, w):
                if not mask[nb]:
                    frontier.add(nb)
        if int(mask.sum()) != target_area:
            continue

        for _ in range(random.randint(0, 3)):
            mask = np.rot90(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)

        mask = _trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 3:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if mask.all():
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=4):
            continue
        return mask.astype(bool)
    return None


def _mirror_axis2(left_obj: dict, right_obj: dict) -> int | None:
    if left_obj["bbox"][0] != right_obj["bbox"][0] or left_obj["bbox"][2] != right_obj["bbox"][2]:
        return None
    if left_obj["mask"].shape != right_obj["mask"].shape:
        return None
    if left_obj["bbox"][3] >= right_obj["bbox"][1]:
        return None
    if not np.array_equal(np.fliplr(left_obj["mask"]), right_obj["mask"]):
        return None

    axis2 = int(left_obj["bbox"][3] + right_obj["bbox"][1])
    reflected = {(int(r), axis2 - int(c)) for r, c in left_obj["coords"]}
    if reflected != right_obj["coord_set"]:
        return None
    return axis2


def _find_pair_edges(objects: list[dict]) -> tuple[list[tuple[int, int, int]], dict[frozenset[int], int]]:
    edges: list[tuple[int, int, int]] = []
    axis_by_pair: dict[frozenset[int], int] = {}
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            left, right = (objects[i], objects[j])
            if left["bbox"][1] > right["bbox"][1]:
                left, right = right, left
            axis2 = _mirror_axis2(left, right)
            if axis2 is None:
                continue
            a = int(left["index"])
            b = int(right["index"])
            key = frozenset((a, b))
            edges.append((a, b, axis2))
            axis_by_pair[key] = axis2
    return edges, axis_by_pair


def _match_all_but_one(
    count: int,
    edges: list[tuple[int, int, int]],
) -> tuple[int, list[tuple[int, int]]] | None:
    adjacency = {i: set() for i in range(count)}
    for i, j, _axis2 in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    def rec(remaining: set[int]) -> list[tuple[int, int]] | None:
        if not remaining:
            return []
        pivot = min(remaining, key=lambda idx: len(adjacency[idx] & remaining))
        options = sorted(adjacency[pivot] & remaining)
        if not options:
            return None
        for partner in options:
            if partner == pivot:
                continue
            sub = rec(remaining - {pivot, partner})
            if sub is not None:
                return [(pivot, partner), *sub]
        return None

    solutions: list[tuple[int, list[tuple[int, int]]]] = []
    for singleton in range(count):
        matched = rec(set(range(count)) - {singleton})
        if matched is not None:
            solutions.append((singleton, matched))
    if len(solutions) != 1:
        return None
    return solutions[0]


def _side_of_axis(obj: dict, axis2: int) -> str:
    if 2 * int(obj["bbox"][3]) < axis2:
        return "left"
    if 2 * int(obj["bbox"][1]) > axis2:
        return "right"
    raise AssertionError("expected the lone object to lie strictly on one side of the exemplar axis")


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    objects_raw = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    assert len(objects_raw) >= 5 and len(objects_raw) % 2 == 1

    objects: list[dict] = []
    for comp in sorted(objects_raw, key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1]))):
        bbox = tuple(map(int, comp["bbox"]))
        coords = np.asarray(comp["coords"], dtype=int)
        mask = crop_bbox(g, bbox) != bg
        obj = {
            "index": len(objects),
            "bbox": bbox,
            "coords": coords,
            "coord_set": {tuple(map(int, rc)) for rc in coords},
            "mask": mask,
            "color": int(comp["color"]),
            "family": _family_key(mask),
        }
        objects.append(obj)

    edges, axis_by_pair = _find_pair_edges(objects)
    matched = _match_all_but_one(len(objects), edges)
    assert matched is not None, "expected exactly one unpaired object"
    lone_idx, pairs = matched
    lone = objects[lone_idx]

    target_pairs = [
        (i, j)
        for i, j in pairs
        if objects[i]["family"] == lone["family"] == objects[j]["family"]
    ]
    assert len(target_pairs) == 1, "expected the lone object to match exactly one mirrored pair by shape"

    a, b = target_pairs[0]
    left_idx, right_idx = (a, b)
    if objects[left_idx]["bbox"][1] > objects[right_idx]["bbox"][1]:
        left_idx, right_idx = right_idx, left_idx
    axis2 = axis_by_pair[frozenset((a, b))]

    side = _side_of_axis(lone, axis2)
    fill_color = objects[right_idx]["color"] if side == "left" else objects[left_idx]["color"]

    out = g.copy()
    new_cells = []
    for r, c in lone["coord_set"]:
        rr = int(r)
        cc = int(axis2 - c)
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
        assert int(out[rr, cc]) == bg, "expected the mirrored partner location to be empty"
        new_cells.append((rr, cc))
    for rr, cc in new_cells:
        out[rr, cc] = int(fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the unique lone sprite by mirroring it across the matching pair's hidden vertical axis and recoloring it with that pair's opposite-side color.

    Input structure:
    - A black background contains several monochrome connected sprites.
    - All but one sprite belong to mirrored left-right pairs: the two members of a pair have the same shape up to horizontal reflection and lie symmetrically around an unseen vertical axis.
    - The two members of a pair use different colors, so pairing must come from shape and position rather than color alone.

    Rule:
    - Find every pair of sprites whose masks are horizontal mirrors of each other and whose absolute positions are symmetric around one vertical axis.
    - The only unpaired sprite shares its shape family with exactly one of those mirrored pairs.
    - Reflect the lone sprite across that pair's axis and paint the reflected copy with the color used by the opposite-side member of the exemplar pair; keep all existing pixels unchanged.

    Key insight:
    - Each mirrored pair teaches two things at once: where the hidden vertical axis is and how color changes when the shape crosses that axis, so the correct completion ignores nearer axes and same-color distractors that do not share the lone sprite's shape.
    """
    return _solve(grid)


def _row_layout(heights: list[int]) -> tuple[int, list[int]] | None:
    gap_count = len(heights) + 1
    min_total = sum(heights) + ROW_GAP_MIN * gap_count
    if min_total > GRID_MAX:
        return None
    height = random.randint(max(18, min_total), GRID_MAX)
    gaps = [ROW_GAP_MIN] * gap_count
    extras = height - min_total
    for _ in range(extras):
        gaps[random.randrange(gap_count)] += 1

    tops: list[int] = []
    cursor = gaps[0]
    for idx, band_h in enumerate(heights):
        tops.append(cursor)
        cursor += band_h + gaps[idx + 1]
    return height, tops


def _choose_shapes(pair_count: int) -> tuple[list[np.ndarray], int] | None:
    shapes: list[np.ndarray] = []
    seen = set()
    for _ in range(pair_count):
        shape = _random_shape()
        if shape is None:
            return None
        family = _family_key(shape)
        if family in seen:
            return None
        seen.add(family)
        shapes.append(shape)
    return shapes, random.randrange(pair_count)


def _place_pairs(
    width: int,
    shapes: list[np.ndarray],
    pair_rows: dict[int, int],
) -> list[dict] | None:
    specs: list[dict] = []
    used_axes: set[int] = set()
    for pair_idx, mask in enumerate(shapes):
        h, w = map(int, mask.shape)
        max_gap = min(PAIR_GAP_RANGE[1], width - 2 * w)
        if max_gap < PAIR_GAP_RANGE[0]:
            return None
        placed = False
        for _ in range(100):
            left_mask = mask if random.random() < 0.5 else np.fliplr(mask)
            right_mask = np.fliplr(left_mask)
            gap = random.randint(PAIR_GAP_RANGE[0], max_gap)
            left_c0 = random.randint(0, width - (2 * w + gap))
            right_c0 = left_c0 + w + gap
            axis2 = int(left_c0 + w - 1 + right_c0)
            if axis2 in used_axes:
                continue
            used_axes.add(axis2)
            specs.append(
                {
                    "pair_idx": pair_idx,
                    "top": int(pair_rows[pair_idx]),
                    "left_mask": left_mask.astype(bool),
                    "right_mask": right_mask.astype(bool),
                    "left_c0": int(left_c0),
                    "right_c0": int(right_c0),
                    "axis2": axis2,
                    "shape": mask.astype(bool),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return specs


def _find_lone_plan(
    width: int,
    pair_specs: list[dict],
    target_idx: int,
    lone_top: int,
) -> dict | None:
    axes = [int(spec["axis2"]) for spec in pair_specs]
    target_spec = pair_specs[target_idx]
    w = int(target_spec["left_mask"].shape[1])
    sides = ["left", "right"]
    random.shuffle(sides)

    candidates: list[dict] = []
    for side in sides:
        for c0 in range(0, width - w + 1):
            mirror_c0 = int(target_spec["axis2"] - (c0 + w - 1))
            if mirror_c0 < 0 or mirror_c0 + w > width:
                continue
            if side == "left":
                if 2 * (c0 + w - 1) >= target_spec["axis2"]:
                    continue
                if mirror_c0 - (c0 + w - 1) < 2:
                    continue
            else:
                if 2 * c0 <= target_spec["axis2"]:
                    continue
                if c0 - (mirror_c0 + w - 1) < 2:
                    continue

            center2 = 2 * c0 + w - 1
            target_dist = abs(int(target_spec["axis2"]) - center2)
            closer = [
                idx
                for idx, axis2 in enumerate(axes)
                if idx != target_idx and abs(axis2 - center2) < target_dist
            ]
            if not closer:
                continue

            closest_decoy = min(closer, key=lambda idx: abs(axes[idx] - center2))
            lone_mask = target_spec["left_mask"] if side == "left" else target_spec["right_mask"]
            candidates.append(
                {
                    "top": int(lone_top),
                    "c0": int(c0),
                    "side": side,
                    "mask": lone_mask.astype(bool),
                    "mirror_c0": mirror_c0,
                    "decoy_idx": int(closest_decoy),
                }
            )
    if not candidates:
        return None
    return random.choice(candidates)


def _assign_pair_colors(pair_count: int, target_idx: int, side: str, decoy_idx: int) -> list[tuple[int, int]]:
    colors: list[tuple[int, int] | None] = [None] * pair_count
    target_left, target_right = random.sample(PALETTE, 2)
    colors[target_idx] = (int(target_left), int(target_right))
    lone_color = int(target_left if side == "left" else target_right)

    decoy_other = random.choice([c for c in PALETTE if c != lone_color])
    if random.random() < 0.5:
        colors[decoy_idx] = (lone_color, int(decoy_other))
    else:
        colors[decoy_idx] = (int(decoy_other), lone_color)

    for idx in range(pair_count):
        if colors[idx] is not None:
            continue
        left_color, right_color = random.sample(PALETTE, 2)
        colors[idx] = (int(left_color), int(right_color))

    return [pair for pair in colors if pair is not None]


def _render_grid(
    height: int,
    width: int,
    pair_specs: list[dict],
    pair_colors: list[tuple[int, int]],
    lone_plan: dict,
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((height, width), BG, dtype=int)
    for spec, (left_color, right_color) in zip(pair_specs, pair_colors):
        paste(grid, _mask_to_sprite(spec["left_mask"], left_color), spec["top"], spec["left_c0"], transparent=BG)
        paste(grid, _mask_to_sprite(spec["right_mask"], right_color), spec["top"], spec["right_c0"], transparent=BG)

    lone_color = pair_colors[target_idx][0] if lone_plan["side"] == "left" else pair_colors[target_idx][1]
    paste(
        grid,
        _mask_to_sprite(lone_plan["mask"], lone_color),
        lone_plan["top"],
        lone_plan["c0"],
        transparent=BG,
    )

    expected = grid.copy()
    partner_color = pair_colors[target_idx][1] if lone_plan["side"] == "left" else pair_colors[target_idx][0]
    paste(
        expected,
        _mask_to_sprite(np.fliplr(lone_plan["mask"]), partner_color),
        lone_plan["top"],
        lone_plan["mirror_c0"],
        transparent=BG,
    )
    return grid, expected


def _valid_scene(
    grid: np.ndarray,
    expected: np.ndarray,
    pair_count: int,
) -> bool:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    if len(comps) != 2 * pair_count + 1:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        pair_count = random.randint(*PAIR_COUNT_RANGE)
        chosen = _choose_shapes(pair_count)
        if chosen is None:
            continue
        shapes, target_idx = chosen

        band_defs = [("pair", idx, int(shape.shape[0])) for idx, shape in enumerate(shapes)]
        band_defs.append(("lone", target_idx, int(shapes[target_idx].shape[0])))
        random.shuffle(band_defs)
        layout = _row_layout([item[2] for item in band_defs])
        if layout is None:
            continue
        height, tops = layout

        pair_rows: dict[int, int] = {}
        lone_top = None
        for (kind, idx, _band_h), top in zip(band_defs, tops):
            if kind == "pair":
                pair_rows[int(idx)] = int(top)
            else:
                lone_top = int(top)
        if lone_top is None:
            continue

        min_width = max(18, max(2 * int(shape.shape[1]) + PAIR_GAP_RANGE[0] for shape in shapes))
        if min_width > GRID_MAX:
            continue
        width = random.randint(min_width, GRID_MAX)

        pair_specs = _place_pairs(width, shapes, pair_rows)
        if pair_specs is None:
            continue
        pair_specs.sort(key=lambda spec: int(spec["pair_idx"]))

        lone_plan = _find_lone_plan(width, pair_specs, target_idx, lone_top)
        if lone_plan is None:
            continue

        pair_colors = _assign_pair_colors(pair_count, target_idx, lone_plan["side"], lone_plan["decoy_idx"])
        grid, expected = _render_grid(height, width, pair_specs, pair_colors, lone_plan, target_idx)

        if _valid_scene(grid, expected, pair_count):
            return grid
    raise RuntimeError("failed to generate a valid mirror-pair ARC instance")
