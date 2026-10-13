from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

TEMPLATE_SIZE = 4
ROTATION_TO_COLOR = {0: RED, 1: BLUE, 2: GREEN, 3: ORANGE}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -int(turns))


def _coord_set(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _translate(coords: frozenset[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in coords)


def _match_translation(
    absolute_group: frozenset[tuple[int, int]], relative_pattern: frozenset[tuple[int, int]]
) -> tuple[int, int] | None:
    if len(absolute_group) != len(relative_pattern):
        return None
    matches = set()
    for gr, gc in absolute_group:
        for pr, pc in relative_pattern:
            dr = gr - pr
            dc = gc - pc
            if _translate(relative_pattern, dr, dc) == absolute_group:
                matches.add((dr, dc))
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _find_divider(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(candidates) == 1, "expected one full grey divider column"
    divider = int(candidates[0])
    assert divider > 0 and divider < grid.shape[1] - 1, "divider must split two non-empty panels"
    assert divider == grid.shape[1] - divider - 1, "panels must have equal width"
    return divider


def _rotations(template: np.ndarray) -> list[np.ndarray]:
    return [_rotate_cw(template, turns) for turns in range(4)]


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "grid must fit ARC size limits"

    divider = _find_divider(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]
    assert left.shape == right.shape, "panels must be equal"

    left_fg = left != BLACK
    left_coords = np.argwhere(left_fg)
    assert left_coords.size > 0, "left panel must contain the template"
    template_box = _bbox(left_coords)
    template = _crop(left, template_box)
    assert template.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE), "template must be 4x4"
    assert np.all(template != BLACK), "template sprite must occupy its full 4x4 window"
    assert set(map(int, np.unique(template))) == {RED, YELLOW}, "template must use only red and yellow"

    marker_mask = template == YELLOW
    marker_coords = np.argwhere(marker_mask)
    assert 0 < len(marker_coords) < TEMPLATE_SIZE * TEMPLATE_SIZE, "template needs marker and non-marker cells"
    assert len(_components(marker_mask)) == 1, "template markers must form one 4-connected pattern"
    mr0, mc0, mr1, mc1 = _bbox(marker_coords)
    assert (mr0, mc0, mr1, mc1) == (0, 0, TEMPLATE_SIZE - 1, TEMPLATE_SIZE - 1), (
        "template markers must touch all four sides so target anchoring is unambiguous"
    )

    rotated_templates = _rotations(template)
    rotated_markers = [_coord_set(np.argwhere(rot == YELLOW)) for rot in rotated_templates]
    assert len(set(rotated_markers)) == 4, "all four marker rotations must be distinct"

    right_colors = set(map(int, np.unique(right)))
    assert right_colors <= {BLACK, YELLOW}, "input target panel may contain only black and yellow markers"
    groups = _components(right == YELLOW)
    assert 2 <= len(groups) <= 4, "target panel must contain 2-4 marker groups"

    matches: list[dict] = []
    for group in groups:
        group_set = _coord_set(group)
        found: list[tuple[int, tuple[int, int]]] = []
        for turns, rel_pattern in enumerate(rotated_markers):
            top_left = _match_translation(group_set, rel_pattern)
            if top_left is None:
                continue
            pr, pc = top_left
            assert 0 <= pr <= right.shape[0] - TEMPLATE_SIZE
            assert 0 <= pc <= right.shape[1] - TEMPLATE_SIZE
            found.append((turns, top_left))
        assert len(found) == 1, "each target group must match exactly one template rotation"
        turns, (top_r, top_c) = found[0]
        matches.append(
            {
                "coords": group,
                "theta": turns,
                "top_left": (int(top_r), int(top_c)),
            }
        )

    thetas = [match["theta"] for match in matches]
    assert len(set(thetas)) >= 2, "at least two different target rotations must appear"

    return {
        "grid": g,
        "divider": divider,
        "left": left,
        "right": right,
        "template": template,
        "template_box": template_box,
        "rotated_templates": rotated_templates,
        "matches": matches,
    }


def _render_output(info: dict) -> np.ndarray:
    g = info["grid"].copy()
    divider = info["divider"]
    out_right = g[:, divider + 1 :].copy()
    planned: dict[tuple[int, int], int] = {}

    for match in info["matches"]:
        turns = int(match["theta"])
        patch = info["rotated_templates"][turns].copy()
        patch[patch == RED] = ROTATION_TO_COLOR[turns]
        top_r, top_c = match["top_left"]
        for r in range(TEMPLATE_SIZE):
            for c in range(TEMPLATE_SIZE):
                color = int(patch[r, c])
                if color == BLACK:
                    continue
                key = (top_r + r, top_c + c)
                previous = planned.get(key)
                assert previous is None or previous == color, "stamps must not conflict"
                planned[key] = color

    for (r, c), color in planned.items():
        existing = int(out_right[r, c])
        assert existing in {BLACK, YELLOW, color}, "target stamps may only overlay their own markers"
        out_right[r, c] = color

    g[:, divider + 1 :] = out_right
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the left-panel template onto right-panel marker groups and recolor the stamp body by rotation.

    Input structure:
    - A black background grid is split into two equal-height panels by a full grey vertical divider.
    - The left panel contains one complete 4x4 template sprite made of yellow marker pixels and red non-marker pixels.
    - The right panel contains 2-4 yellow marker groups whose relative layouts equal rotated copies of the template's yellow pattern.

    Rule:
    - Extract the 4x4 template and enumerate its 0, 90, 180, and 270 degree clockwise rotations.
    - For each right-panel marker group, find the unique rotation whose yellow coordinates match the group up to translation.
    - Stamp that rotated full template onto the group location; keep yellow marker pixels yellow and recolor the rotated red cells by angle:
      0 degrees -> red, 90 degrees -> blue, 180 degrees -> green, 270 degrees -> orange.
    - Preserve the left panel and the grey divider unchanged.

    Key insight:
    - The same rotation inference is the bottleneck for both outputs: it determines where the full template goes and which body color the stamped copy must use.
    """
    return _render_output(_analyze_input(grid))


def _random_connected_marker_mask() -> np.ndarray:
    for _ in range(500):
        target_size = random.randint(5, 8)
        mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
        seed = (random.randrange(TEMPLATE_SIZE), random.randrange(TEMPLATE_SIZE))
        cells = {seed}
        mask[seed] = True
        while len(cells) < target_size:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < TEMPLATE_SIZE and 0 <= nc < TEMPLATE_SIZE and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nxt = random.choice(tuple(frontier))
            cells.add(nxt)
            mask[nxt] = True

        if len(cells) != target_size:
            continue
        coords = np.argwhere(mask)
        if len(_components(mask)) != 1:
            continue
        if _bbox(coords) != (0, 0, TEMPLATE_SIZE - 1, TEMPLATE_SIZE - 1):
            continue
        rotations = [_coord_set(np.argwhere(_rotate_cw(mask.astype(int), k) == 1)) for k in range(4)]
        if len(set(rotations)) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate a valid marker mask")


def _place_targets(panel_shape: tuple[int, int], count: int) -> list[tuple[int, int]]:
    h, w = panel_shape
    candidates = [(r, c) for r in range(h - TEMPLATE_SIZE + 1) for c in range(w - TEMPLATE_SIZE + 1)]

    def far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
        ar0, ac0 = a
        br0, bc0 = b
        ar1, ac1 = ar0 + TEMPLATE_SIZE - 1, ac0 + TEMPLATE_SIZE - 1
        br1, bc1 = br0 + TEMPLATE_SIZE - 1, bc0 + TEMPLATE_SIZE - 1
        row_gap = max(0, max(ar0, br0) - min(ar1, br1) - 1)
        col_gap = max(0, max(ac0, bc0) - min(ac1, bc1) - 1)
        return row_gap >= 2 or col_gap >= 2

    for _ in range(500):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for pos in candidates:
            if all(far_enough(pos, other) for other in chosen):
                chosen.append(pos)
                if len(chosen) == count:
                    return chosen
    raise RuntimeError("failed to place target groups")


def _choose_rotations(count: int) -> list[int]:
    if count == 2:
        return random.sample(range(4), 2)
    if count == 3:
        return random.sample(range(4), 3)
    return random.sample(range(4), 4)


def _build_instance() -> np.ndarray:
    target_count = random.choice([2, 3, 4])
    panel_h = random.choice([10, 11, 12, 13, 14])
    panel_w = random.choice([10, 11, 12, 13, 14])

    grid = np.zeros((panel_h, 2 * panel_w + 1), dtype=int)
    divider = panel_w
    grid[:, divider] = GREY

    marker_mask = _random_connected_marker_mask()
    template = np.where(marker_mask, YELLOW, RED).astype(int)

    left_top = (random.randrange(panel_h - TEMPLATE_SIZE + 1), random.randrange(panel_w - TEMPLATE_SIZE + 1))
    lr, lc = left_top
    grid[lr : lr + TEMPLATE_SIZE, lc : lc + TEMPLATE_SIZE] = template

    target_positions = _place_targets((panel_h, panel_w), target_count)
    rotations = _choose_rotations(target_count)
    random.shuffle(rotations)
    for (top_r, top_c), turns in zip(target_positions, rotations):
        rot_mask = _rotate_cw(marker_mask.astype(int), turns).astype(bool)
        panel_c = divider + 1 + top_c
        patch = np.where(rot_mask, YELLOW, BLACK).astype(int)
        existing = grid[top_r : top_r + TEMPLATE_SIZE, panel_c : panel_c + TEMPLATE_SIZE]
        assert not np.any((patch != BLACK) & (existing != BLACK))
        existing[patch == YELLOW] = YELLOW

    return grid


def _validate_generated_input(grid: np.ndarray) -> None:
    info = _analyze_input(grid)
    out = main(grid)

    assert np.array_equal(out, _render_output(info)), "main must reproduce the analyzed stamping plan"
    assert set(map(int, np.unique(grid[:, info["divider"] : info["divider"] + 1]))) == {GREY}
    assert np.array_equal(out[:, : info["divider"]], grid[:, : info["divider"]]), "left panel must stay unchanged"

    changed = np.argwhere(out != grid)
    assert len(changed) > 0, "output must add stamped pixels"

    right_in = grid[:, info["divider"] + 1 :]
    right_out = out[:, info["divider"] + 1 :]
    for match in info["matches"]:
        turns = int(match["theta"])
        color = ROTATION_TO_COLOR[turns]
        top_r, top_c = match["top_left"]
        in_patch = right_in[top_r : top_r + TEMPLATE_SIZE, top_c : top_c + TEMPLATE_SIZE]
        out_patch = right_out[top_r : top_r + TEMPLATE_SIZE, top_c : top_c + TEMPLATE_SIZE]
        rot_template = info["rotated_templates"][turns]
        marker_mask = rot_template == YELLOW
        body_mask = rot_template == RED
        assert np.array_equal(in_patch[marker_mask], np.full(marker_mask.sum(), YELLOW)), "input must expose only markers"
        assert np.all(in_patch[body_mask] == BLACK), "input body cells must be empty before stamping"
        assert np.all(out_patch[marker_mask] == YELLOW), "markers must remain yellow"
        assert np.all(out_patch[body_mask] == color), "body recolor must follow the matched rotation"

    assert any(match["theta"] != 0 for match in info["matches"]), "at least one target must require recoloring"


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_instance()
        try:
            _validate_generated_input(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
