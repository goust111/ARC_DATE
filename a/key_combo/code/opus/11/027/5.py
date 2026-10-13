import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RAY_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
GRID_SIDE_RANGE = (16, 22)
SPAN_RANGE = (3, 5)
DEPTH_RANGE = (2, 4)
PAIR_GAP_RANGE = (4, 8)
PAIR_MARGIN = 1
PAIR_BOX_GAP = 2
MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append({"color": color, "cells": cells})
    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _cup_dimensions(orientation: str, span: int, depth: int) -> tuple[int, int]:
    if orientation in ("up", "down"):
        return depth + 1, span + 1
    return span + 1, depth + 1


def _opening_positions(side_len: int, opening_len: int) -> list[int]:
    return list(range(1, side_len - opening_len))


def _expected_cup_mask(height: int, width: int, orientation: str, opening_start: int, opening_len: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if orientation == "up":
        mask[0, opening_start : opening_start + opening_len] = False
    elif orientation == "down":
        mask[-1, opening_start : opening_start + opening_len] = False
    elif orientation == "left":
        mask[opening_start : opening_start + opening_len, 0] = False
    elif orientation == "right":
        mask[opening_start : opening_start + opening_len, -1] = False
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    return mask


def _opening_local_coords(height: int, width: int, orientation: str, opening_start: int, opening_len: int) -> list[tuple[int, int]]:
    if orientation == "up":
        return [(0, c) for c in range(opening_start, opening_start + opening_len)]
    if orientation == "down":
        return [(height - 1, c) for c in range(opening_start, opening_start + opening_len)]
    if orientation == "left":
        return [(r, 0) for r in range(opening_start, opening_start + opening_len)]
    if orientation == "right":
        return [(r, width - 1) for r in range(opening_start, opening_start + opening_len)]
    raise ValueError(f"unknown orientation: {orientation}")


def _identify_cup(component: dict) -> dict:
    cells = component["cells"]
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True

    matches = []
    for orientation in ("up", "down", "left", "right"):
        side_len = width if orientation in ("up", "down") else height
        for opening_len in range(1, side_len - 1):
            for opening_start in _opening_positions(side_len, opening_len):
                expected = _expected_cup_mask(height, width, orientation, opening_start, opening_len)
                if np.array_equal(mask, expected):
                    matches.append((orientation, opening_start, opening_len))
    assert len(matches) == 1, "each component must be a unique cup with one opening side"
    orientation, opening_start, opening_len = matches[0]
    opening_cells = [(r0 + dr, c0 + dc) for dr, dc in _opening_local_coords(height, width, orientation, opening_start, opening_len)]
    cavity_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return {
        "color": component["color"],
        "cells": cells,
        "bbox": (r0, c0, r1, c1),
        "height": height,
        "width": width,
        "orientation": orientation,
        "direction": RAY_DIR[orientation],
        "opening_start": opening_start,
        "opening_len": opening_len,
        "opening_cells": opening_cells,
        "cavity_cells": cavity_cells,
    }


def _owner_map(shape: tuple[int, int], cups: list[dict]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for idx, cup in enumerate(cups):
        for r, c in cup["cells"]:
            owner[r, c] = idx
    return owner


def _trace_rays(grid: np.ndarray, cups: list[dict]) -> list[dict]:
    owner = _owner_map(grid.shape, cups)
    enriched = []
    for idx, cup in enumerate(cups):
        dr, dc = cup["direction"]
        target_ids = []
        ray_cells = []
        ray_lengths = []
        for start_r, start_c in cup["opening_cells"]:
            r = start_r
            c = start_c
            path = []
            hit = None
            while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                other = int(owner[r, c])
                if other != -1:
                    hit = other
                    break
                path.append((r, c))
                r += dr
                c += dc
            assert hit is not None and hit != idx, "each opening ray must meet one opposite cup"
            target_ids.append(hit)
            ray_cells.extend(path)
            ray_lengths.append(len(path))
        assert len(set(target_ids)) == 1, "all opening pixels of one cup must target the same opposite cup"
        cup2 = dict(cup)
        cup2["target"] = target_ids[0]
        cup2["ray_cells"] = ray_cells
        cup2["ray_lengths"] = ray_lengths
        enriched.append(cup2)
    return enriched


def _analyze_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    components = _find_components(g)
    assert len(components) == 4, "expected exactly four cups"
    cups = [_identify_cup(component) for component in components]
    return _trace_rays(g, cups)


def main(grid: np.ndarray) -> np.ndarray:
    """Extend mouth rays from each cup, using the color of the opposite cup that the ray hits.

    Input structure:
    - The grid has black background and exactly four monochrome cup-like containers.
    - Each cup is a one-cell-thick rectangular cup with a contiguous opening on exactly one side.
    - The cups form two facing pairs, and every opening direction sees exactly one opposite cup across empty space.

    Rule:
    - Detect each cup's opening side and the opening cells on that side.
    - From every opening cell, extend a straight ray through black cells until the ray would enter the first opposite cup body.
    - Color the whole ray with the color of that opposite cup, not the color of the emitting cup.
    - Keep all cup pixels unchanged and leave every cup cavity black.

    Key insight:
    - The visible beams are keyed by the receiver: you must pair cups geometrically before you know the ray color.
    """
    g = np.asarray(grid, dtype=int)
    cups = _analyze_grid(g)
    planned = {}
    for cup in cups:
        target_color = cups[cup["target"]]["color"]
        for cell in cup["ray_cells"]:
            previous = planned.get(cell)
            assert previous is None or previous == target_color, "ray colors must not conflict"
            planned[cell] = target_color

    out = g.copy()
    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def _cup_cells_from_params(
    r0: int,
    c0: int,
    orientation: str,
    span: int,
    depth: int,
    opening_start: int,
    opening_len: int,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    height, width = _cup_dimensions(orientation, span, depth)
    mask = _expected_cup_mask(height, width, orientation, opening_start, opening_len)
    cells = [(r0 + r, c0 + c) for r, c in np.argwhere(mask)]
    return cells, (r0, c0, r0 + height - 1, c0 + width - 1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _disjoint_intervals(side_len: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    options = []
    for opening_len in range(1, min(3, side_len - 2) + 1):
        for opening_start in _opening_positions(side_len, opening_len):
            cells = set(range(opening_start, opening_start + opening_len))
            options.append((opening_start, opening_len, cells))
    random.shuffle(options)
    for start_a, len_a, cells_a in options:
        others = options[:]
        random.shuffle(others)
        for start_b, len_b, cells_b in others:
            if cells_a.isdisjoint(cells_b):
                return (start_a, len_a), (start_b, len_b)
    return None


def _build_pair_template(kind: str, colors: list[int]) -> dict | None:
    gap = random.randint(*PAIR_GAP_RANGE)
    span = random.randint(*SPAN_RANGE)
    if kind == "H":
        depth_a = random.randint(*DEPTH_RANGE)
        depth_b = random.randint(*DEPTH_RANGE)
        height, width_a = _cup_dimensions("right", span, depth_a)
        height2, width_b = _cup_dimensions("left", span, depth_b)
        assert height == height2
        openings = _disjoint_intervals(height)
        if openings is None:
            return None
        (start_a, len_a), (start_b, len_b) = openings
        return {
            "kind": kind,
            "box_h": height,
            "box_w": width_a + gap + width_b,
            "cups": [
                {
                    "orientation": "right",
                    "span": span,
                    "depth": depth_a,
                    "opening_start": start_a,
                    "opening_len": len_a,
                    "color": colors[0],
                    "offset": (0, 0),
                },
                {
                    "orientation": "left",
                    "span": span,
                    "depth": depth_b,
                    "opening_start": start_b,
                    "opening_len": len_b,
                    "color": colors[1],
                    "offset": (0, width_a + gap),
                },
            ],
        }
    if kind == "V":
        depth_a = random.randint(*DEPTH_RANGE)
        depth_b = random.randint(*DEPTH_RANGE)
        height_a, width = _cup_dimensions("down", span, depth_a)
        height_b, width2 = _cup_dimensions("up", span, depth_b)
        assert width == width2
        openings = _disjoint_intervals(width)
        if openings is None:
            return None
        (start_a, len_a), (start_b, len_b) = openings
        return {
            "kind": kind,
            "box_h": height_a + gap + height_b,
            "box_w": width,
            "cups": [
                {
                    "orientation": "down",
                    "span": span,
                    "depth": depth_a,
                    "opening_start": start_a,
                    "opening_len": len_a,
                    "color": colors[0],
                    "offset": (0, 0),
                },
                {
                    "orientation": "up",
                    "span": span,
                    "depth": depth_b,
                    "opening_start": start_b,
                    "opening_len": len_b,
                    "color": colors[1],
                    "offset": (height_a + gap, 0),
                },
            ],
        }
    raise ValueError(f"unknown pair kind: {kind}")


def _place_template(template: dict, origin: tuple[int, int]) -> list[dict]:
    base_r, base_c = origin
    placed = []
    for cup in template["cups"]:
        dr, dc = cup["offset"]
        cells, bbox = _cup_cells_from_params(
            base_r + dr,
            base_c + dc,
            cup["orientation"],
            cup["span"],
            cup["depth"],
            cup["opening_start"],
            cup["opening_len"],
        )
        placed.append(
            {
                "orientation": cup["orientation"],
                "span": cup["span"],
                "depth": cup["depth"],
                "opening_start": cup["opening_start"],
                "opening_len": cup["opening_len"],
                "color": cup["color"],
                "cells": cells,
                "bbox": bbox,
            }
        )
    return placed


def _center_signature(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + r1, c0 + c1


def _render_grid(height: int, width: int, cups: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for cup in cups:
        for r, c in cup["cells"]:
            assert 0 <= r < height and 0 <= c < width
            assert grid[r, c] == BACKGROUND
            grid[r, c] = cup["color"]
    return grid


def _random_origins(height: int, width: int, templates: list[dict]) -> list[tuple[int, int]] | None:
    placed_boxes = []
    stored = []
    order = list(range(len(templates)))
    random.shuffle(order)
    for idx in order:
        template = templates[idx]
        candidates = []
        for r0 in range(PAIR_MARGIN, height - template["box_h"] - PAIR_MARGIN + 2):
            for c0 in range(PAIR_MARGIN, width - template["box_w"] - PAIR_MARGIN + 2):
                box = (r0, c0, r0 + template["box_h"] - 1, c0 + template["box_w"] - 1)
                if all(_boxes_separated(box, other, PAIR_BOX_GAP) for other in placed_boxes):
                    candidates.append((r0, c0, box))
        if not candidates:
            return None
        r0, c0, box = random.choice(candidates)
        stored.append((idx, (r0, c0)))
        placed_boxes.append(box)
    origin_map = {idx: origin for idx, origin in stored}
    return [origin_map[i] for i in range(len(templates))]


def _validate_generated_grid(grid: np.ndarray) -> None:
    cups = _analyze_grid(grid)
    assert len({cup["color"] for cup in cups}) == 4

    row_centers = Counter()
    col_centers = Counter()
    painted = {}
    for idx, cup in enumerate(cups):
        row_sig, col_sig = _center_signature(cup["bbox"])
        row_centers[row_sig] += 1
        col_centers[col_sig] += 1
        partner = cups[cup["target"]]
        assert partner["target"] == idx, "pairing must be reciprocal"
        assert partner["orientation"] == OPPOSITE[cup["orientation"]]
        assert partner["color"] != cup["color"]
        assert all(length >= 4 for length in cup["ray_lengths"]), "rays should be visually non-trivial"
        assert len(set(cup["ray_lengths"])) == 1, "parallel rays from one opening should stop together"
        for cell in cup["ray_cells"]:
            assert cell not in painted, "opposite-color rays must stay disjoint"
            painted[cell] = idx

    assert max(row_centers.values()) <= 2, "no three cups may share the same center row"
    assert max(col_centers.values()) <= 2, "no three cups may share the same center column"

    out = main(grid)
    assert np.any(out != grid), "output must differ from input"
    for cup in cups:
        for r, c in cup["cavity_cells"]:
            assert int(out[r, c]) == BACKGROUND, "cup cavities must remain black"


def _generation_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    pair_kinds = random.choice([["H", "H"], ["H", "V"], ["V", "V"]])
    colors = random.sample(PALETTE, 4)
    templates = []
    for pair_kind, pair_colors in zip(pair_kinds, (colors[:2], colors[2:])):
        template = _build_pair_template(pair_kind, pair_colors)
        if template is None:
            return None
        templates.append(template)
    origins = _random_origins(height, width, templates)
    if origins is None:
        return None

    cups = []
    for template, origin in zip(templates, origins):
        cups.extend(_place_template(template, origin))
    grid = _render_grid(height, width, cups)
    try:
        _validate_generated_grid(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
