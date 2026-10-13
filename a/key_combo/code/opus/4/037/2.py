from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, get_bg  # noqa: E402


BG = Color.BLACK
CRACK = Color.MAROON
MERGED = Color.TEAL
BODY_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE)

HORIZONTAL_CRACK_TEMPLATES = (
    np.array(
        [
            [1, 0],
            [1, 1],
            [9, 9],
            [1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1],
            [9, 9],
            [1, 1],
            [0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1],
            [1, 1],
            [9, 9],
            [1, 0],
        ],
        dtype=int,
    ),
)

VERTICAL_CRACK_TEMPLATES = (
    np.array(
        [
            [1, 9, 1, 1],
            [1, 9, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 9, 1],
            [1, 0, 9, 1],
        ],
        dtype=int,
    ),
)

SUPPORT_TEMPLATE = np.array(
    [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0],
    ],
    dtype=int,
)

ADJACENT_TEMPLATE = np.array(
    [
        [1, 0],
        [1, 1],
        [9, 9],
        [1, 1],
    ],
    dtype=int,
)

BLOCKED_TEMPLATE = np.array(
    [
        [1, 0, 0],
        [1, 1, 0],
        [9, 9, 9],
        [0, 1, 1],
    ],
    dtype=int,
)

SCENE_HEIGHT = 15
SCENE_WIDTH = 8
BASE_SCENE_ITEMS = (
    ("blocked", BLOCKED_TEMPLATE, 0, 5),
    ("adjacent", ADJACENT_TEMPLATE, 1, 0),
    ("support", SUPPORT_TEMPLATE, 12, 1),
)


def _colorize_template(template: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros_like(template, dtype=int)
    patch[template == 1] = int(color)
    patch[template == 9] = int(CRACK)
    return patch


def _body_coords(component_coords: np.ndarray, grid: np.ndarray, color: int) -> np.ndarray:
    rows = component_coords[:, 0]
    cols = component_coords[:, 1]
    body_mask = grid[rows, cols] == int(color)
    return component_coords[body_mask]


def _split_cracked_component(component: dict, grid: np.ndarray, source_id: int) -> list[dict]:
    coords = np.asarray(component["coords"], dtype=int)
    colors = {int(grid[r, c]) for r, c in coords}
    body_color = next(iter(colors - {CRACK}))
    crack_coords = coords[grid[coords[:, 0], coords[:, 1]] == int(CRACK)]
    body = _body_coords(coords, grid, body_color)

    unique_rows = sorted({int(r) for r, _ in crack_coords})
    unique_cols = sorted({int(c) for _, c in crack_coords})
    parts: list[dict] = []

    if len(unique_cols) == 1:
        crack_col = unique_cols[0]
        left = body[body[:, 1] < crack_col] + np.array([0, -1], dtype=int)
        right = body[body[:, 1] > crack_col] + np.array([0, 1], dtype=int)
        assert left.size and right.size
        for role, shifted in (("left", left), ("right", right)):
            parts.append(
                {
                    "kind": "fragment",
                    "source_id": source_id,
                    "source_color": int(body_color),
                    "source_orientation": "vertical",
                    "role": role,
                    "color": int(body_color),
                    "coords": shifted,
                    "initial_bbox": bbox_coords(shifted),
                }
            )
        return parts

    assert len(unique_rows) == 1
    crack_row = unique_rows[0]
    top = body[body[:, 0] < crack_row] + np.array([-1, 0], dtype=int)
    bottom = body[body[:, 0] > crack_row] + np.array([1, 0], dtype=int)
    assert top.size and bottom.size
    for role, shifted in (("top", top), ("bottom", bottom)):
        parts.append(
            {
                "kind": "fragment",
                "source_id": source_id,
                "source_color": int(body_color),
                "source_orientation": "horizontal",
                "role": role,
                "color": int(body_color),
                "coords": shifted,
                "initial_bbox": bbox_coords(shifted),
            }
        )
    return parts


def _extract_participants(grid: np.ndarray) -> tuple[list[dict], dict[int, list[int]]]:
    components = find_components(grid, bg=BG, connectivity=8, monochromatic=False)
    participants: list[dict] = []
    source_members: dict[int, list[int]] = {}
    source_id = 0
    object_id = 0

    for component in components:
        coords = np.asarray(component["coords"], dtype=int)
        colors = {int(grid[r, c]) for r, c in coords}
        if CRACK in colors:
            assert len(colors) == 2
            part_list = _split_cracked_component(component, grid, source_id)
            source_members[source_id] = []
            for part in part_list:
                part["object_id"] = object_id
                source_members[source_id].append(len(participants))
                participants.append(part)
            source_id += 1
        else:
            assert len(colors) == 1
            color = next(iter(colors))
            participants.append(
                {
                    "kind": "whole",
                    "source_id": None,
                    "source_color": int(color),
                    "source_orientation": None,
                    "role": "whole",
                    "color": int(color),
                    "coords": coords.copy(),
                    "initial_bbox": bbox_coords(coords),
                    "object_id": object_id,
                }
            )
        object_id += 1
    return participants, source_members


def _gravity_key(participant: dict) -> tuple[int, int, int]:
    r0, c0, r1, _c1 = participant["initial_bbox"]
    return (-int(r1), -int(r0), int(c0))


def _settle_participants(grid_shape: tuple[int, int], participants: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    h, w = map(int, grid_shape)
    out = np.full((h, w), int(BG), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    owner = np.full((h, w), -1, dtype=int)
    settled = [dict(participant) for participant in participants]

    for index in sorted(range(len(settled)), key=lambda idx: _gravity_key(settled[idx])):
        coords = np.asarray(settled[index]["coords"], dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        shift = 0
        while True:
            next_rows = rows + shift + 1
            if np.any(next_rows >= h):
                break
            if np.any(occupied[next_rows, cols]):
                break
            shift += 1

        final_coords = np.column_stack((rows + shift, cols))
        settled[index]["index"] = index
        settled[index]["final_coords"] = final_coords
        settled[index]["final_bbox"] = bbox_coords(final_coords)
        out[final_coords[:, 0], final_coords[:, 1]] = int(settled[index]["color"])
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
        owner[final_coords[:, 0], final_coords[:, 1]] = index

    return out, owner, settled


def _vertically_adjacent(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    cells_b = {tuple(map(int, cell)) for cell in coords_b}
    for r, c in coords_a:
        if (int(r) + 1, int(c)) in cells_b or (int(r) - 1, int(c)) in cells_b:
            return True
    return False


def _pair_info(first: dict, second: dict, owner: np.ndarray) -> dict:
    coords_a = np.asarray(first["final_coords"], dtype=int)
    coords_b = np.asarray(second["final_coords"], dtype=int)
    adjacent = _vertically_adjacent(coords_a, coords_b)

    cols_a = defaultdict(list)
    cols_b = defaultdict(list)
    for r, c in coords_a:
        cols_a[int(c)].append(int(r))
    for r, c in coords_b:
        cols_b[int(c)].append(int(r))

    overlap_cols = sorted(set(cols_a) & set(cols_b))
    blocked_by_other = False

    if overlap_cols:
        upper, lower = (first, second) if first["final_bbox"][0] <= second["final_bbox"][0] else (second, first)
        upper_cols = cols_a if upper is first else cols_b
        lower_cols = cols_b if lower is second else cols_a
        for col in overlap_cols:
            upper_bottom = max(upper_cols[col])
            lower_top = min(lower_cols[col])
            if upper_bottom + 1 >= lower_top:
                continue
            for row in range(upper_bottom + 1, lower_top):
                owner_id = int(owner[row, col])
                if owner_id >= 0 and owner_id not in {int(first["index"]), int(second["index"])}:
                    blocked_by_other = True
                    break
            if blocked_by_other:
                break

    return {
        "source_id": int(first["source_id"]),
        "adjacent": adjacent,
        "blocked_by_other": blocked_by_other,
        "overlap_cols": overlap_cols,
        "members": (int(first["index"]), int(second["index"])),
    }


def _solve(grid: np.ndarray, return_state: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == int(BG)

    participants, source_members = _extract_participants(g)
    out, owner, settled = _settle_participants(g.shape, participants)

    pair_infos = []
    for source_id, member_ids in source_members.items():
        assert len(member_ids) == 2
        first = settled[member_ids[0]]
        second = settled[member_ids[1]]
        info = _pair_info(first, second, owner)
        pair_infos.append(info)
        if info["adjacent"]:
            for participant in (first, second):
                coords = participant["final_coords"]
                out[coords[:, 0], coords[:, 1]] = int(MERGED)

    if not return_state:
        return out

    return out, {
        "participants": settled,
        "owner": owner,
        "pair_infos": pair_infos,
        "source_members": source_members,
        "object_members": _group_members_by_object(settled),
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Split every cracked object into two offset halves, let all resulting pieces fall under shared gravity, and recolor sibling halves that end up vertically touching.

    Input structure:
    - A black background contains 3 or 4 separate 8-connected foreground objects.
    - Each object has its own body color from blue, green, yellow, or orange; some objects also contain an internal maroon crack line.
    - A crack line fully spans the object's shorter bbox axis: wide objects use a vertical crack, tall objects use a horizontal crack.

    Rule:
    - Erase each maroon crack line and split that object into two halves by shifting the two sides one cell away from the crack along the crack axis.
    - Treat every resulting fragment and every uncracked object as an independent rigid body, and drop them downward in order from lower original position to higher original position.
    - After settling, inspect every sibling fragment pair from the same original object; if the two fragments are vertically adjacent in at least one shared column, recolor both fragments to teal.
    - All other settled objects and fragments keep their original body colors.

    Key insight:
    - The crack changes the set of bodies that participate in gravity, and only the full gravity simulation reveals whether sibling fragments end up touching or remain separated by blockers.
    """
    return _solve(input_grid, return_state=False)


def _flip_patch(patch: np.ndarray) -> np.ndarray:
    return np.fliplr(patch) if random.random() < 0.5 else patch.copy()


def _group_members_by_object(participants: list[dict]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for participant in participants:
        grouped[int(participant["object_id"])].append(int(participant["index"]))
    return dict(grouped)


def _can_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False

    cells = np.argwhere(patch != 0)
    for pr, pc in cells:
        rr = top + int(pr)
        cc = left + int(pc)
        if grid[rr, cc] != 0:
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if np.any(grid[r0:r1, c0:c1] != 0):
            return False
    return True


def _stamp(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    cells = np.argwhere(patch != 0)
    for pr, pc in cells:
        grid[top + int(pr), left + int(pc)] = int(patch[pr, pc])


def _non_square_non_rectangular(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    b = bbox_coords(coords)
    h = b[2] - b[0] + 1
    w = b[3] - b[1] + 1
    return h != w and int(mask.sum()) < h * w


def _component_summary(component: dict, grid: np.ndarray) -> dict:
    coords = np.asarray(component["coords"], dtype=int)
    colors = {int(grid[r, c]) for r, c in coords}
    summary = {
        "area": int(coords.shape[0]),
        "bbox": tuple(map(int, component["bbox"])),
        "colors": colors,
    }
    if CRACK in colors:
        crack = coords[grid[coords[:, 0], coords[:, 1]] == int(CRACK)]
        summary["body_color"] = int(next(iter(colors - {CRACK})))
        summary["crack_rows"] = sorted({int(r) for r, _ in crack})
        summary["crack_cols"] = sorted({int(c) for _, c in crack})
    else:
        summary["body_color"] = int(next(iter(colors)))
        summary["crack_rows"] = []
        summary["crack_cols"] = []
    return summary


def _pair_adjacent_after_removing_object(state: dict, grid_shape: tuple[int, int], source_id: int) -> bool:
    target_members = set(state["source_members"][source_id])

    for object_id, member_ids in state["object_members"].items():
        if target_members & set(member_ids):
            continue

        reduced = []
        for participant in state["participants"]:
            if int(participant["object_id"]) != int(object_id):
                reduced.append(
                    {
                        "kind": participant["kind"],
                        "source_id": participant["source_id"],
                        "source_color": participant["source_color"],
                        "source_orientation": participant["source_orientation"],
                        "role": participant["role"],
                        "color": participant["color"],
                        "coords": np.asarray(participant["coords"], dtype=int).copy(),
                        "initial_bbox": tuple(map(int, participant["initial_bbox"])),
                        "object_id": int(participant["object_id"]),
                    }
                )

        _, _, resettled = _settle_participants(grid_shape, reduced)
        siblings = [participant for participant in resettled if participant["source_id"] == source_id]
        if len(siblings) == 2 and _vertically_adjacent(siblings[0]["final_coords"], siblings[1]["final_coords"]):
            return True

    return False


def _validate_input_structure(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if get_bg(grid) != int(BG):
        return False

    components = find_components(grid, bg=BG, connectivity=8, monochromatic=False)
    if not (3 <= len(components) <= 4):
        return False

    body_colors = set()
    cracked = 0
    tops = set()

    for component in components:
        summary = _component_summary(component, grid)
        r0, c0, r1, c1 = summary["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        coords = np.asarray(component["coords"], dtype=int)
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if summary["area"] < 4 or summary["area"] > 8:
            return False
        if not _non_square_non_rectangular(mask):
            return False
        tops.add(r0)
        body_color = summary["body_color"]
        if body_color in body_colors:
            return False
        body_colors.add(body_color)
        if body_color not in BODY_COLORS:
            return False

        if CRACK in summary["colors"]:
            cracked += 1
            if len(summary["colors"]) != 2:
                return False
            if len(summary["crack_cols"]) == 1:
                if w <= h:
                    return False
                if len(summary["crack_rows"]) != h:
                    return False
            elif len(summary["crack_rows"]) == 1:
                if h <= w:
                    return False
                if len(summary["crack_cols"]) != w:
                    return False
            else:
                return False
        elif len(summary["colors"]) != 1:
            return False

    return cracked in {2, 3} and len(tops) == len(components)


def _validate_semantics(grid: np.ndarray) -> bool:
    if not _validate_input_structure(grid):
        return False

    out, state = _solve(grid, return_state=True)
    if np.array_equal(out, grid):
        return False

    pair_infos = state["pair_infos"]
    if len(pair_infos) not in {2, 3}:
        return False

    adjacent_pairs = [info for info in pair_infos if info["adjacent"]]
    if len(adjacent_pairs) != 1:
        return False

    blocked_pairs = [
        info
        for info in pair_infos
        if (not info["adjacent"]) and _pair_adjacent_after_removing_object(state, grid.shape, int(info["source_id"]))
    ]
    if not blocked_pairs:
        return False

    recolored = int(np.sum(out == int(MERGED)))
    if recolored == 0:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    include_extra = random.random() < 0.55
    height = random.randint(18, 24)
    width = random.randint(14, 18 if include_extra else 16)
    grid = np.zeros((height, width), dtype=int)

    color_order = list(BODY_COLORS)
    random.shuffle(color_order)

    mirror_scene = random.random() < 0.5
    base_width = SCENE_WIDTH
    extra_template = random.choice(VERTICAL_CRACK_TEMPLATES)
    extra_width = extra_template.shape[1]
    gap = 2

    if include_extra:
        extra_on_left = random.random() < 0.5
        if extra_on_left:
            scene_left_min = extra_width + gap + 1
            scene_left_max = width - base_width
        else:
            scene_left_min = 0
            scene_left_max = width - (base_width + gap + extra_width + 1)
        if scene_left_min > scene_left_max:
            return None
        scene_left = random.randint(scene_left_min, scene_left_max)
        extra_left_choices = (
            list(range(1, scene_left - gap - extra_width + 1))
            if extra_on_left
            else list(range(scene_left + base_width + gap, width - extra_width))
        )
        if not extra_left_choices:
            return None
        extra_left = random.choice(extra_left_choices)
    else:
        scene_left = random.randint(0, width - base_width)
        extra_left = None

    scene_top = random.randint(1, height - SCENE_HEIGHT)

    scene_colors = {
        "blocked": color_order.pop(),
        "adjacent": color_order.pop(),
        "support": color_order.pop(),
    }

    for name, template, local_top, local_left in BASE_SCENE_ITEMS:
        patch = template.copy()
        left = int(local_left)
        if mirror_scene:
            patch = np.fliplr(patch)
            left = SCENE_WIDTH - patch.shape[1] - left
        patch = _colorize_template(patch, scene_colors[name])
        top = scene_top + int(local_top)
        left = scene_left + left
        if not _can_place(grid, patch, top, left):
            return None
        _stamp(grid, patch, top, left)

    if include_extra:
        extra_patch = _colorize_template(_flip_patch(extra_template), color_order.pop())
        top_choices = list(range(0, height - extra_patch.shape[0] + 1))
        random.shuffle(top_choices)
        placed = False
        for top in top_choices:
            if _can_place(grid, extra_patch, top, int(extra_left)):
                _stamp(grid, extra_patch, top, int(extra_left))
                placed = True
                break
        if not placed:
            return None

    return grid if _validate_semantics(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fractured-gravity ARC input")
