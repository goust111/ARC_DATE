from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, neighbors4


BAR_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE)
SIDES = ("top", "right", "bottom", "left")
OVERLAP_QUADRANTS = {
    "NE": (("top", 1), ("right", -1)),
    "NW": (("top", -1), ("left", -1)),
    "SE": (("bottom", 1), ("right", 1)),
    "SW": (("bottom", -1), ("left", 1)),
}
MAX_GENERATION_ATTEMPTS = 400


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _bar_length(comp: dict) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(comp["area"])
    assert h == 1 or w == 1, "expected one-cell-thick control bars"
    assert area == max(h, w), "expected a solid straight bar"
    return area


def _contact_side(bar_coords: np.ndarray, template_set: set[tuple[int, int]]) -> str:
    touched_sides: set[str] = set()
    for r, c in bar_coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if (int(nr), int(nc)) not in template_set:
                continue
            if rr == nr - 1:
                touched_sides.add("top")
            elif rr == nr + 1:
                touched_sides.add("bottom")
            elif cc == nc - 1:
                touched_sides.add("left")
            elif cc == nc + 1:
                touched_sides.add("right")
    assert len(touched_sides) == 1, "each bar must touch exactly one template side"
    return next(iter(touched_sides))


def _bar_step(template_bbox: tuple[int, int, int, int], bar_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    tr, tc = _bbox_center(template_bbox)
    br, bc = _bbox_center(bar_bbox)
    dr = int(np.sign(br - tr))
    dc = int(np.sign(bc - tc))
    assert dr != 0 or dc != 0
    return dr, dc


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    template_candidates = [comp for comp in components if int(comp["color"]) == Color.GREY]
    assert len(template_candidates) == 1, "expected one gray template"
    template = template_candidates[0]
    assert 4 <= int(template["area"]) <= 8

    template_bbox = tuple(map(int, template["bbox"]))
    template_crop = crop_bbox(g, template_bbox)
    template_mask = template_crop == Color.GREY
    h, w = template_mask.shape
    assert h <= 4 and w <= 4

    template_set = {tuple(map(int, rc)) for rc in template["coords"]}

    bars = []
    for comp in components:
        if int(comp["color"]) == Color.GREY:
            continue
        color = int(comp["color"])
        assert color in BAR_COLORS
        side = _contact_side(comp["coords"], template_set)
        bars.append(
            {
                "color": color,
                "coords": comp["coords"],
                "bbox": tuple(map(int, comp["bbox"])),
                "length": _bar_length(comp),
                "side": side,
            }
        )

    assert 2 <= len(bars) <= 4
    assert len({bar["color"] for bar in bars}) == len(bars)
    assert len({bar["length"] for bar in bars}) == len(bars)
    assert len({bar["side"] for bar in bars}) == len(bars)

    bars.sort(key=lambda bar: (bar["bbox"][0], bar["bbox"][1], bar["color"]))
    for bar in bars:
        bar["step"] = _bar_step(template_bbox, bar["bbox"])

    for i, bar0 in enumerate(bars):
        cells0 = [tuple(map(int, rc)) for rc in bar0["coords"]]
        for bar1 in bars[i + 1 :]:
            cells1 = [tuple(map(int, rc)) for rc in bar1["coords"]]
            for r0, c0 in cells0:
                for r1, c1 in cells1:
                    assert abs(r0 - r1) + abs(c0 - c1) > 1, "bars must not share an edge"

    return {
        "grid": g,
        "bg": bg,
        "template_bbox": template_bbox,
        "template_mask": template_mask,
        "template_set": template_set,
        "bars": bars,
    }


def _winning_stamps(scene: dict) -> tuple[dict[tuple[int, int], tuple[tuple[int, int], int]], dict[int, set[tuple[int, int]]], list[tuple[int, int]]]:
    g = scene["grid"]
    r0, c0, _r1, _c1 = scene["template_bbox"]
    h, w = scene["template_mask"].shape

    best: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    winners_by_bar: dict[int, set[tuple[int, int]]] = {idx: set() for idx in range(len(scene["bars"]))}
    overlap_cells: list[tuple[int, int]] = []

    seen_candidates: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    template_set = scene["template_set"]
    for order, bar in enumerate(scene["bars"]):
        dr, dc = bar["step"]
        for k in range(1, bar["length"] + 1):
            top = r0 + k * dr * h
            left = c0 + k * dc * w
            for rr, cc in np.argwhere(scene["template_mask"]):
                gr = int(top + rr)
                gc = int(left + cc)
                assert 0 <= gr < g.shape[0] and 0 <= gc < g.shape[1], "generated copy fell outside the grid"
                if (gr, gc) in template_set:
                    continue
                seen_candidates.setdefault((gr, gc), []).append((bar["length"], order, bar["color"]))
                priority = (bar["length"], order)
                current = best.get((gr, gc))
                if current is None or priority < current[0]:
                    if current is not None:
                        previous_order = current[0][1]
                        winners_by_bar[previous_order].discard((gr, gc))
                    best[(gr, gc)] = (priority, bar["color"])
                    winners_by_bar[order].add((gr, gc))

    for cell, candidates in seen_candidates.items():
        if len(candidates) >= 2:
            overlap_cells.append(cell)

    return best, winners_by_bar, overlap_cells


def _render_output(grid: np.ndarray) -> np.ndarray:
    scene = _parse_scene(grid)
    out = np.asarray(grid, dtype=int).copy()
    for bar in scene["bars"]:
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = scene["bg"]

    best, _winners_by_bar, _overlap_cells = _winning_stamps(scene)
    for (r, c), (_priority, color) in best.items():
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp repeated recolored copies of a gray template along the outward direction encoded by each touching colored bar.

    Input structure:
    - The grid has a black background, one gray 4-connected template sprite, and two to four one-cell-thick colored bars.
    - Each bar touches a different template side and each bar has a distinct color and distinct length.
    - The template fits inside a small bbox, while every bar extends outward from the side it touches.

    Rule:
    - Extract the gray template bbox and keep the original gray template in place.
    - For each touching bar, decode an outward step from the bar center relative to the template center and repeat the template once per bar pixel.
    - Each repeated copy moves by one template-bbox height vertically and one template-bbox width horizontally in the decoded step direction, and gray template pixels are recolored to the bar color.
    - Remove the bars themselves; when copies overlap, the shorter bar's copies win, with stable earlier parsing order breaking ties.

    Key insight:
    - Every bar simultaneously encodes a translation direction, a repetition count, and a layer priority for overlapping stamped copies.
    """
    return _render_output(np.asarray(grid, dtype=int))


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        h = random.choices((2, 3, 4), weights=(5, 4, 2), k=1)[0]
        w = random.choices((2, 3, 4), weights=(5, 4, 2), k=1)[0]
        area = random.randint(4, min(8, h * w))

        cells = {(random.randrange(h), random.randrange(w))}
        frontier = set()
        while len(cells) < area:
            for r, c in tuple(cells):
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True

        coords = np.argwhere(mask)
        if coords.shape[0] != area:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
        if not (4 <= int(cropped.sum()) <= 8):
            continue
        if cropped.shape[0] > 4 or cropped.shape[1] > 4:
            continue
        return cropped
    raise RuntimeError("failed to generate a connected template mask")


def _anchor_choices(mask: np.ndarray) -> dict[str, list[int]]:
    return {
        "top": [int(c) for c in np.flatnonzero(mask[0])],
        "bottom": [int(c) for c in np.flatnonzero(mask[-1])],
        "left": [int(r) for r in np.flatnonzero(mask[:, 0])],
        "right": [int(r) for r in np.flatnonzero(mask[:, -1])],
    }


def _orth_sign(side: str, anchor: int, mask_shape: tuple[int, int]) -> int:
    h, w = mask_shape
    if side in {"top", "bottom"}:
        center = (w - 1) / 2.0
    else:
        center = (h - 1) / 2.0
    return int(np.sign(anchor - center))


def _step_from_side(side: str, anchor: int, mask_shape: tuple[int, int]) -> tuple[int, int]:
    orth = _orth_sign(side, anchor, mask_shape)
    if side == "top":
        return -1, orth
    if side == "bottom":
        return 1, orth
    if side == "left":
        return orth, -1
    return orth, 1


def _bar_cells(side: str, anchor: int, length: int, r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0 - 1 - i, c0 + anchor) for i in range(length)]
    if side == "bottom":
        return [(r0 + h + i, c0 + anchor) for i in range(length)]
    if side == "left":
        return [(r0 + anchor, c0 - 1 - i) for i in range(length)]
    return [(r0 + anchor, c0 + w + i) for i in range(length)]


def _bars_separated(bars: list[dict]) -> bool:
    for i, bar0 in enumerate(bars):
        for bar1 in bars[i + 1 :]:
            for r0, c0 in bar0["cells"]:
                for r1, c1 in bar1["cells"]:
                    if abs(r0 - r1) + abs(c0 - c1) <= 1:
                        return False
    return True


def _choose_bar_specs(mask: np.ndarray, require_overlap: bool) -> list[dict] | None:
    choices = _anchor_choices(mask)
    bar_count = random.choices((2, 3, 4), weights=(5, 4, 2), k=1)[0]
    bars: list[dict] = []

    used_sides: set[str] = set()
    used_vectors: dict[tuple[int, int], int] = {}

    if require_overlap:
        quadrants = list(OVERLAP_QUADRANTS.items())
        random.shuffle(quadrants)
        pair = None
        for _name, ((side_a, orth_a), (side_b, orth_b)) in quadrants:
            anchors_a = [a for a in choices[side_a] if _orth_sign(side_a, a, mask.shape) == orth_a]
            anchors_b = [a for a in choices[side_b] if _orth_sign(side_b, a, mask.shape) == orth_b]
            if anchors_a and anchors_b:
                pair = ((side_a, random.choice(anchors_a)), (side_b, random.choice(anchors_b)))
                break
        if pair is None:
            return None
        for idx, (side, anchor) in enumerate(pair):
            step = _step_from_side(side, anchor, mask.shape)
            bars.append(
                {
                    "side": side,
                    "anchor": int(anchor),
                    "step": step,
                }
            )
            used_sides.add(side)
            used_vectors[step] = used_vectors.get(step, 0) + 1

    for idx in range(len(bars), bar_count):
        candidates = []
        for side in SIDES:
            if side in used_sides:
                continue
            for anchor in choices[side]:
                step = _step_from_side(side, int(anchor), mask.shape)
                if require_overlap:
                    if used_vectors.get(step, 0) >= 2:
                        continue
                else:
                    if step in used_vectors:
                        continue
                candidates.append((side, int(anchor), step))
        if not candidates:
            return None
        side, anchor, step = random.choice(candidates)
        bars.append(
            {
                "side": side,
                "anchor": anchor,
                "step": step,
            }
        )
        used_sides.add(side)
        used_vectors[step] = used_vectors.get(step, 0) + 1

    if require_overlap and max(used_vectors.values()) < 2:
        return None

    h, w = mask.shape
    for bar in bars:
        dr, dc = bar["step"]
        cost = 0
        if dr != 0:
            cost += h
        if dc != 0:
            cost += w
        if bar["side"] in {"top", "bottom"}:
            cost += 1
        if bar["side"] in {"left", "right"}:
            cost += 1
        bar["cost"] = cost

    length_sets = list(itertools.combinations((2, 3, 4, 5), bar_count))
    weights = [1.0 / sum(lengths) for lengths in length_sets]
    chosen_lengths = list(random.choices(length_sets, weights=weights, k=1)[0])
    chosen_lengths.sort()
    bars_by_cost = sorted(range(len(bars)), key=lambda idx: (-bars[idx]["cost"], bars[idx]["side"]))
    for length, idx in zip(chosen_lengths, bars_by_cost):
        bars[idx]["length"] = int(length)

    colors = random.sample(BAR_COLORS, bar_count)
    random.shuffle(colors)
    for color, bar in zip(colors, bars):
        bar["color"] = int(color)
        del bar["cost"]
    return bars


def _required_margins(mask: np.ndarray, bars: list[dict]) -> tuple[int, int, int, int]:
    h, w = mask.shape
    top = bottom = left = right = 1
    for bar in bars:
        length = int(bar["length"])
        side = bar["side"]
        dr, dc = bar["step"]
        if side == "top":
            top = max(top, length + 1)
        elif side == "bottom":
            bottom = max(bottom, length + 1)
        elif side == "left":
            left = max(left, length + 1)
        else:
            right = max(right, length + 1)

        if dr < 0:
            top = max(top, length * h + 1)
        elif dr > 0:
            bottom = max(bottom, length * h + 1)
        if dc < 0:
            left = max(left, length * w + 1)
        elif dc > 0:
            right = max(right, length * w + 1)
    return top, bottom, left, right


def _build_input(mask: np.ndarray, bars: list[dict]) -> np.ndarray | None:
    h, w = mask.shape
    top_need, bottom_need, left_need, right_need = _required_margins(mask, bars)

    min_h = top_need + h + bottom_need
    min_w = left_need + w + right_need
    if min_h > 30 or min_w > 30:
        return None

    grid_h = min_h + random.randint(0, min(6, 30 - min_h))
    grid_w = min_w + random.randint(0, min(6, 30 - min_w))
    r0 = top_need + random.randint(0, grid_h - min_h)
    c0 = left_need + random.randint(0, grid_w - min_w)

    for bar in bars:
        bar["cells"] = _bar_cells(bar["side"], bar["anchor"], bar["length"], r0, c0, h, w)

    if not _bars_separated(bars):
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for rr, cc in np.argwhere(mask):
        grid[r0 + int(rr), c0 + int(cc)] = Color.GREY

    for bar in bars:
        for rr, cc in bar["cells"]:
            if not (0 <= rr < grid_h and 0 <= cc < grid_w):
                return None
            if grid[rr, cc] != Color.BLACK:
                return None
            grid[rr, cc] = int(bar["color"])
    return grid


def _check_semantics(grid: np.ndarray, require_overlap: bool) -> bool:
    try:
        scene = _parse_scene(grid)
        expected = _render_output(grid)
    except AssertionError:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    best, winners_by_bar, overlap_cells = _winning_stamps(scene)
    if any(len(cells) == 0 for cells in winners_by_bar.values()):
        return False

    if require_overlap:
        if not overlap_cells:
            return False
        demonstrated = False
        for r, c in overlap_cells:
            candidates = []
            for order, bar in enumerate(scene["bars"]):
                dr, dc = bar["step"]
                h, w = scene["template_mask"].shape
                tr0, tc0, _tr1, _tc1 = scene["template_bbox"]
                for k in range(1, bar["length"] + 1):
                    top = tr0 + k * dr * h
                    left = tc0 + k * dc * w
                    if 0 <= r - top < h and 0 <= c - left < w and scene["template_mask"][r - top, c - left]:
                        candidates.append((bar["length"], order, bar["color"]))
            if len(candidates) >= 2:
                winner = min(candidates)
                if best[(r, c)] == ((winner[0], winner[1]), winner[2]):
                    demonstrated = True
                    break
        if not demonstrated:
            return False
    else:
        if overlap_cells and random.random() < 0.5:
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    mask = _random_connected_mask()
    for _ in range(80):
        require_overlap = random.random() < 0.7
        bars = _choose_bar_specs(mask, require_overlap=require_overlap)
        if bars is None:
            continue
        top, bottom, left, right = _required_margins(mask, bars)
        if top + mask.shape[0] + bottom > 30 or left + mask.shape[1] + right > 30:
            continue
        grid = _build_input(mask, bars)
        if grid is None:
            continue
        if not _check_semantics(grid, require_overlap=require_overlap):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
