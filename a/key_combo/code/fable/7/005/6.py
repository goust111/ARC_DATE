from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import Color, bbox, bbox_h, crop_bbox, find_components, get_bg


BACKGROUND = Color.BLACK
WALL = Color.RED
SEPARATOR = Color.GREY
POINTER = Color.PINK
BAR_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.TEAL, Color.MAROON]
RESERVED_COLORS = {BACKGROUND, WALL, SEPARATOR, POINTER}

DIR_BY_NAME = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
NAME_BY_DIR = {value: key for key, value in DIR_BY_NAME.items()}
OPPOSITE = {
    DIR_BY_NAME["up"]: DIR_BY_NAME["down"],
    DIR_BY_NAME["down"]: DIR_BY_NAME["up"],
    DIR_BY_NAME["left"]: DIR_BY_NAME["right"],
    DIR_BY_NAME["right"]: DIR_BY_NAME["left"],
}

MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _shift_coords(coords: list[tuple[int, int]], dr: int, dc: int, steps: int = 1) -> list[tuple[int, int]]:
    return [(int(r) + int(dr) * int(steps), int(c) + int(dc) * int(steps)) for r, c in coords]


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top) + int(r), int(left) + int(c)) for r, c in np.argwhere(mask)]


def _side_cells(mask: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if dr == -1:
        edge = min(r for r, _ in cells)
        return [(r, c) for r, c in cells if r == edge]
    if dr == 1:
        edge = max(r for r, _ in cells)
        return [(r, c) for r, c in cells if r == edge]
    if dc == -1:
        edge = min(c for _, c in cells)
        return [(r, c) for r, c in cells if c == edge]
    edge = max(c for _, c in cells)
    return [(r, c) for r, c in cells if c == edge]


def _simulate_motion(
    coords: list[tuple[int, int]],
    direction: tuple[int, int],
    steps: int,
    walls: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], str]:
    dr, dc = map(int, direction)
    current = list(coords)
    reason = "full"
    for _ in range(int(steps)):
        nxt = _shift_coords(current, dr, dc)
        if any((r, c) in walls for r, c in nxt):
            reason = "wall"
            break
        if any(not _in_bounds(shape, r, c) for r, c in nxt):
            reason = "edge"
            break
        current = nxt
    return current, reason


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == SEPARATOR))]
    assert len(cols) == 1, "expected one full-height grey separator column"
    return int(cols[0])


def _bar_heights(left: np.ndarray, bg: int) -> dict[int, int]:
    heights: dict[int, int] = {}
    for color in sorted(set(map(int, left.flatten())) - {int(bg)}):
        assert color not in RESERVED_COLORS, "legend bars must avoid reserved colors"
        comps = find_components(left, bg=bg, connectivity=4, monochromatic=True, colors=[color])
        assert len(comps) == 1, "each legend color must form one bar"
        comp = comps[0]
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert r1 == left.shape[0] - 1, "bars must be bottom-aligned"
        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch == color)), "bars must be solid rectangles"
        b = bbox(left == color)
        assert b is not None
        heights[int(color)] = int(bbox_h(b))
    assert len(heights) >= 1
    assert len(set(heights.values())) == len(heights), "bar heights must be unique"
    return heights


def _extract_sprites(right: np.ndarray, bg: int, heights: dict[int, int]) -> list[dict]:
    sprite_colors = sorted(heights)
    comps = find_components(
        right,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=sprite_colors,
        return_format="dict",
    )
    pink_coords = [tuple(map(int, rc)) for rc in np.argwhere(right == POINTER)]
    used_pointers: set[tuple[int, int]] = set()
    sprites = []
    for comp in sorted(comps, key=lambda item: (int(item["color"]), tuple(map(int, item["bbox"])))):
        coord_set = {tuple(map(int, rc)) for rc in comp["coords"]}
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for pr, pc in pink_coords:
            touched = []
            for dr, dc in DIR_BY_NAME.values():
                if (pr - dr, pc - dc) in coord_set:
                    touched.append((int(dr), int(dc)))
            if touched:
                assert len(touched) == 1, "pointer must touch exactly one sprite edge cell"
                candidates.append(((int(pr), int(pc)), touched[0]))
        assert len(candidates) == 1, "each sprite must have exactly one pink pointer"
        pointer, direction = candidates[0]
        assert pointer not in used_pointers, "each pointer must belong to one sprite"
        used_pointers.add(pointer)
        sprites.append(
            {
                "color": int(comp["color"]),
                "coords": sorted(coord_set),
                "pointer": pointer,
                "direction": direction,
                "step_count": int(heights[int(comp["color"])]),
            }
        )
    assert sprites, "right side must contain at least one sprite"
    assert used_pointers == set(pink_coords), "all pink pixels must be sprite pointers"
    return sprites


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))
    sep = _separator_col(g)
    h, w = map(int, g.shape)
    left = g[:, :sep]
    right = crop_bbox(g, (0, sep + 1, h - 1, w - 1))
    heights = _bar_heights(left, bg)
    walls = {tuple(map(int, rc)) for rc in np.argwhere(right == WALL)}
    sprites = _extract_sprites(right, bg, heights)
    return {
        "bg": bg,
        "separator": sep,
        "left": left,
        "right": right,
        "heights": heights,
        "walls": walls,
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read left-side bar heights as a color legend and march the right-side sprites along their pink pointers until they finish, hit a red wall, or reach the edge.

    Input structure:
    - A full-height grey separator column splits the canvas into a left legend and a right action field on a black background.
    - The left side contains several bottom-aligned solid vertical bars whose colors are all distinct and whose heights are all distinct.
    - The right side contains monochrome sprites in those legend colors, exactly one pink adjacent pointer pixel per sprite, and scattered red wall cells.

    Rule:
    - Measure each bar height and treat it as the step count for sprites of the same color.
    - Discard the separator and the entire left legend, keeping only the right field size.
    - For every sprite, move the whole shape in the direction from the sprite toward its pink pointer by that many steps, but stop early at the last legal position before a red wall or before leaving the grid.

    Key insight:
    - The answer is determined only by combining three separate clues: legend color-to-height mapping, pointer direction, and wall or boundary truncation on the right side.
    """
    scene = _parse_scene(grid)
    right = scene["right"]
    bg = int(scene["bg"])
    walls = set(scene["walls"])

    out = np.full(right.shape, bg, dtype=int)
    for r, c in walls:
        out[r, c] = WALL

    for sprite in scene["sprites"]:
        final_coords, _reason = _simulate_motion(
            sprite["coords"],
            sprite["direction"],
            sprite["step_count"],
            walls,
            right.shape,
        )
        for r, c in final_coords:
            assert out[r, c] == bg, "moved sprites must not overlap walls or each other"
            out[r, c] = int(sprite["color"])
    return out


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target = random.randint(3, min(6, h * w))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        while len(cells) < target:
            frontier.clear()
            for r, c in cells:
                for dr, dc in DIR_BY_NAME.values():
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if mask.shape[0] < 2 and mask.shape[1] < 2:
            continue
        if int(mask.sum()) == mask.size:
            if random.random() < 0.7:
                continue
        return mask
    raise RuntimeError("failed to sample a connected sprite mask")


def _sample_categories(count: int) -> list[str]:
    categories = ["full", random.choice(["wall", "edge"])]
    if count >= 3 and random.random() < 0.75:
        third = "wall" if "wall" not in categories else "edge"
        categories.append(third)
    while len(categories) < count:
        categories.append(random.choice(["full", "wall", "edge"]))
    random.shuffle(categories)
    assert "full" in categories
    assert len(set(categories)) >= 2
    return categories[:count]


def _sample_bar_layout(count: int, heights: list[int]) -> tuple[int, list[dict]]:
    order = list(range(count))
    random.shuffle(order)
    widths = [random.randint(1, 2) for _ in range(count)]
    gaps = [random.randint(1, 2) for _ in range(max(0, count - 1))]
    bars = []
    cursor = 1
    for idx, spec_index in enumerate(order):
        width = widths[idx]
        bars.append(
            {
                "spec_index": int(spec_index),
                "left": int(cursor),
                "width": int(width),
                "height": int(heights[spec_index]),
            }
        )
        cursor += width
        if idx < count - 1:
            cursor += gaps[idx]
    left_width = int(cursor + 1)
    return left_width, bars


def _sample_final_origin(category: str, direction: tuple[int, int], field_h: int, field_w: int, mask: np.ndarray) -> tuple[int, int]:
    h, w = map(int, mask.shape)
    dr, dc = map(int, direction)
    if category == "edge":
        if dr == -1:
            return 0, random.randint(0, field_w - w)
        if dr == 1:
            return field_h - h, random.randint(0, field_w - w)
        if dc == -1:
            return random.randint(0, field_h - h), 0
        return random.randint(0, field_h - h), field_w - w
    return random.randint(0, field_h - h), random.randint(0, field_w - w)


def _place_one_sprite(
    spec: dict,
    field_h: int,
    field_w: int,
    input_occ: set[tuple[int, int]],
    output_occ: set[tuple[int, int]],
) -> dict | None:
    mask = spec["mask"]
    direction = spec["direction"]
    dr, dc = map(int, direction)
    side_cells = _side_cells(mask, direction)
    travel = int(spec["travel"])

    for _ in range(PLACEMENT_ATTEMPTS):
        final_top, final_left = _sample_final_origin(spec["category"], direction, field_h, field_w, mask)
        final_cells = _coords_from_mask(mask, final_top, final_left)
        init_cells = _shift_coords(final_cells, -dr, -dc, travel)
        if any(not _in_bounds((field_h, field_w), r, c) for r, c in init_cells):
            continue
        next_cells = _shift_coords(final_cells, dr, dc)
        if spec["category"] == "edge" and all(_in_bounds((field_h, field_w), r, c) for r, c in next_cells):
            continue

        final_set = set(final_cells)
        if final_set & output_occ:
            continue

        wall_cell: tuple[int, int] | None = None
        if spec["category"] == "wall":
            anchors = side_cells[:]
            random.shuffle(anchors)
            for ar, ac in anchors:
                candidate = (int(final_top) + int(ar) + dr, int(final_left) + int(ac) + dc)
                if _in_bounds((field_h, field_w), *candidate) and candidate not in final_set and candidate not in output_occ:
                    wall_cell = candidate
                    break
            if wall_cell is None:
                continue

        init_set = set(init_cells)
        if init_set & input_occ:
            continue

        anchors = side_cells[:]
        random.shuffle(anchors)
        for ar, ac in anchors:
            pointer = (
                int(final_top) + int(ar) - dr * (travel - 1),
                int(final_left) + int(ac) - dc * (travel - 1),
            )
            if not _in_bounds((field_h, field_w), *pointer):
                continue
            if pointer in init_set or pointer in input_occ:
                continue
            neighbors = {
                (pointer[0] - 1, pointer[1]),
                (pointer[0] + 1, pointer[1]),
                (pointer[0], pointer[1] - 1),
                (pointer[0], pointer[1] + 1),
            }
            if len(neighbors & init_set) != 1:
                continue
            if neighbors & input_occ:
                continue
            if wall_cell is not None and pointer == wall_cell:
                continue

            if wall_cell is not None and wall_cell in input_occ:
                continue

            plan = dict(spec)
            plan["final_cells"] = sorted(final_set)
            plan["init_cells"] = sorted(init_set)
            plan["pointer"] = pointer
            plan["walls"] = [] if wall_cell is None else [wall_cell]
            return plan
    return None


def _all_path_cells(sprite: dict) -> set[tuple[int, int]]:
    cells = set()
    for step in range(1, int(sprite["height"]) + 1):
        cells.update(_shift_coords(sprite["init_cells"], *sprite["direction"], steps=step))
    return cells


def _assemble_right_field(field_h: int, field_w: int, plan: list[dict], extra_walls: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((field_h, field_w), BACKGROUND, dtype=int)
    out = np.full((field_h, field_w), BACKGROUND, dtype=int)
    walls = set(extra_walls)
    for sprite in plan:
        walls.update(sprite["walls"])

    for r, c in walls:
        inp[r, c] = WALL
        out[r, c] = WALL

    for sprite in plan:
        color = int(sprite["color"])
        for r, c in sprite["init_cells"]:
            assert inp[r, c] == BACKGROUND
            inp[r, c] = color
        pr, pc = sprite["pointer"]
        assert inp[pr, pc] == BACKGROUND
        inp[pr, pc] = POINTER
        for r, c in sprite["final_cells"]:
            assert out[r, c] == BACKGROUND
            out[r, c] = color

    return inp, out


def _assemble_input(field_h: int, left_width: int, bars: list[dict], plan: list[dict], right_inp: np.ndarray) -> np.ndarray:
    grid = np.full((field_h, left_width + 1 + right_inp.shape[1]), BACKGROUND, dtype=int)
    grid[:, left_width] = SEPARATOR
    for bar in bars:
        sprite = plan[int(bar["spec_index"])]
        height = int(bar["height"])
        width = int(bar["width"])
        left = int(bar["left"])
        top = field_h - height
        grid[top:, left : left + width] = int(sprite["color"])
    grid[:, left_width + 1 :] = right_inp
    return grid


def _add_extra_walls(field_h: int, field_w: int, plan: list[dict]) -> set[tuple[int, int]]:
    existing = set()
    forbidden = set()
    for sprite in plan:
        existing.update(sprite["init_cells"])
        existing.update(sprite["final_cells"])
        existing.update(sprite["walls"])
        existing.add(tuple(sprite["pointer"]))
        forbidden.update(_all_path_cells(sprite))

    need_wall = "wall" not in {sprite["category"] for sprite in plan}
    extra_count = random.randint(1 if need_wall else 0, 3)
    candidates = [
        (r, c)
        for r in range(field_h)
        for c in range(field_w)
        if (r, c) not in existing and (r, c) not in forbidden
    ]
    random.shuffle(candidates)
    return set(candidates[:extra_count])


def _validate_plan(plan: list[dict], field_h: int, field_w: int, expected_out: np.ndarray) -> None:
    categories = [sprite["category"] for sprite in plan]
    assert "full" in categories
    assert len(set(categories)) >= 2

    walls = set()
    for sprite in plan:
        walls.update(sprite["walls"])

    final_occ: set[tuple[int, int]] = set()
    height_sensitive = 0
    for sprite in plan:
        final_coords, reason = _simulate_motion(
            sprite["init_cells"],
            sprite["direction"],
            sprite["height"],
            walls,
            (field_h, field_w),
        )
        assert set(final_coords) == set(sprite["final_cells"])
        assert reason == sprite["category"]
        assert len(set(sprite["init_cells"])) == len(sprite["init_cells"])
        assert len(set(sprite["final_cells"])) == len(sprite["final_cells"])
        assert final_occ.isdisjoint(sprite["final_cells"])
        final_occ.update(sprite["final_cells"])

        pointer = tuple(sprite["pointer"])
        touch_count = 0
        for dr, dc in DIR_BY_NAME.values():
            if (pointer[0] - dr, pointer[1] - dc) in set(sprite["init_cells"]):
                touch_count += 1
        assert touch_count == 1

        alt_coords, _ = _simulate_motion(
            sprite["init_cells"],
            OPPOSITE[sprite["direction"]],
            sprite["height"],
            walls,
            (field_h, field_w),
        )
        assert set(alt_coords) != set(sprite["final_cells"])

        if sprite["height"] > 1:
            alt_h = sprite["height"] - 1
            shorter_coords, _ = _simulate_motion(
                sprite["init_cells"],
                sprite["direction"],
                alt_h,
                walls,
                (field_h, field_w),
            )
            if set(shorter_coords) != set(sprite["final_cells"]):
                height_sensitive += 1

    assert height_sensitive >= 1
    assert expected_out.ndim == 2 and max(expected_out.shape) <= 30


def _validate_generated_input(grid: np.ndarray) -> dict:
    scene = _parse_scene(grid)
    out = main(grid)
    reasons = []
    final_occ: set[tuple[int, int]] = set()
    for sprite in scene["sprites"]:
        final_coords, reason = _simulate_motion(
            sprite["coords"],
            sprite["direction"],
            sprite["step_count"],
            scene["walls"],
            scene["right"].shape,
        )
        reasons.append(reason)
        assert final_occ.isdisjoint(final_coords)
        final_occ.update(final_coords)
    assert "full" in reasons
    assert len(set(reasons)) >= 2
    assert not np.any(out == POINTER)
    assert np.array_equal(out[scene["right"] == WALL], np.full(int(np.sum(scene["right"] == WALL)), WALL))
    return {
        "sprite_count": len(scene["sprites"]),
        "reasons": reasons,
        "shape": out.shape,
    }


def _generate_one() -> np.ndarray:
    field_h = random.randint(15, 20)
    sprite_count = random.randint(2, 4)
    max_height = min(8, field_h - 5)
    assert max_height >= sprite_count + 1
    heights = random.sample(list(range(2, max_height + 1)), sprite_count)
    categories = _sample_categories(sprite_count)
    colors = random.sample(BAR_COLORS, sprite_count)

    left_width, bars = _sample_bar_layout(sprite_count, heights)
    min_right_width = max(12, max(heights) + 4)
    max_right_width = 29 - left_width
    if min_right_width > max_right_width:
        raise RuntimeError("insufficient width budget")
    field_w = random.randint(min_right_width, min(max_right_width, 18))

    specs = []
    for color, height, category in zip(colors, heights, categories):
        direction_name = random.choice(list(DIR_BY_NAME))
        direction = DIR_BY_NAME[direction_name]
        travel = int(height if category == "full" else random.randint(1, height - 1))
        specs.append(
            {
                "color": int(color),
                "height": int(height),
                "category": category,
                "direction": direction,
                "direction_name": direction_name,
                "travel": travel,
                "mask": _random_connected_mask(),
            }
        )

    specs.sort(key=lambda item: ({"edge": 0, "wall": 1, "full": 2}[item["category"]], -item["height"]))

    placed: list[dict] = []
    input_occ: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()
    for spec in specs:
        placed_sprite = _place_one_sprite(spec, field_h, field_w, input_occ, output_occ)
        if placed_sprite is None:
            raise RuntimeError("failed to place sprite")
        placed.append(placed_sprite)
        input_occ.update(placed_sprite["init_cells"])
        input_occ.add(tuple(placed_sprite["pointer"]))
        input_occ.update(placed_sprite["walls"])
        output_occ.update(placed_sprite["final_cells"])
        output_occ.update(placed_sprite["walls"])

    extra_walls = _add_extra_walls(field_h, field_w, placed)
    right_inp, expected_out = _assemble_right_field(field_h, field_w, placed, extra_walls)
    plan_by_color = {sprite["color"]: sprite for sprite in placed}
    ordered_plan = [plan_by_color[color] for color in colors]
    grid = _assemble_input(field_h, left_width, bars, ordered_plan, right_inp)

    _validate_plan(ordered_plan, field_h, field_w, expected_out)
    actual_out = main(grid)
    assert np.array_equal(actual_out, expected_out)
    _validate_generated_input(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _generate_one()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid Measure March puzzle input")
