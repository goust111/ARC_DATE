from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    if int(connectivity) == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _find_blue_start(grid: np.ndarray) -> int:
    blue = np.argwhere(grid == BLUE)
    assert blue.shape == (1, 2), "expected exactly one blue start pixel"
    start_r, start_c = map(int, blue[0])
    assert start_r == grid.shape[0] - 1, "blue start pixel must be on the bottom row"
    return start_c


def _ray_trace(shape: tuple[int, int], start_c: int) -> list[tuple[int, int, int]]:
    height, width = map(int, shape)
    r = height - 1
    c = int(start_c)
    dc = 1
    states: list[tuple[int, int, int]] = []
    while r > 0:
        next_c = c + dc
        if next_c < 0 or next_c >= width:
            dc *= -1
            next_c = c + dc
        r -= 1
        c = next_c
        states.append((r, c, dc))
    return states


def _in_rect(rect: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _first_direction_in_rect(rect: BBox, states: list[tuple[int, int, int]]) -> int | None:
    for r, c, dc in states:
        if _in_rect(rect, r, c):
            return int(dc)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey block by the travel direction of an invisible bouncing diagonal ray.

    Input structure:
    - A black background contains exactly one blue start pixel on the bottom row.
    - Several grey 4-connected solid blocks are scattered away from the left and right borders.

    Rule:
    - Start an invisible ray at the blue pixel and move it one cell up and one cell right per step.
    - When the ray would leave through the left or right border, reverse only its horizontal direction and keep tracing until it reaches the top row.
    - Any grey block the ray enters is recolored as a whole on first entry: red when the ray is traveling up-right, teal when it is traveling up-left.
    - Blocks never obstruct the ray, the ray itself is not drawn, and untouched grey blocks stay grey.

    Key insight:
    - The output shows only the hidden ray's first-contact direction on each crossed block, so the solver must replay the full reflected path rather than classify blocks by position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    assert 1 < height <= 30 and 1 < width <= 30

    start_c = _find_blue_start(g)
    grey_components = _components_mask(g == GREY, connectivity=4)

    cell_to_component: dict[tuple[int, int], int] = {}
    for idx, coords in enumerate(grey_components):
        r0, c0, r1, c1 = _bbox_coords(coords)
        assert coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "grey components must be solid blocks"
        assert c0 > 0 and c1 < width - 1, "grey blocks must not touch the left or right border"
        for r, c in coords:
            cell_to_component[(int(r), int(c))] = idx

    entry_colors: dict[int, int] = {}
    for r, c, dc in _ray_trace(g.shape, start_c):
        comp_idx = cell_to_component.get((r, c))
        if comp_idx is not None and comp_idx not in entry_colors:
            entry_colors[comp_idx] = RED if dc > 0 else TEAL

    out = g.copy()
    for comp_idx, color in entry_colors.items():
        coords = grey_components[comp_idx]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


# --------------- generator ---------------

MIN_H = 12
MAX_H = 24
MIN_W = 8
MAX_W = 18
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 400


def _rect_conflict(a: BBox, b: BBox, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _build_input(height: int, width: int, start_c: int, rects: list[BBox]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[height - 1, start_c] = BLUE
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return grid


def _candidate_hit_rects(
    height: int, width: int, states: list[tuple[int, int, int]], direction: int
) -> list[dict[str, int | BBox]]:
    candidates: list[dict[str, int | BBox]] = []
    for idx, (r, c, dc) in enumerate(states):
        if dc != direction or r < 2 or r > height - 2:
            continue
        for block_h in range(1, min(3, r) + 1):
            r0 = r - block_h + 1
            r1 = r
            if r0 < 1 or r1 >= height - 1:
                continue
            for block_w in range(1, 4):
                for c0 in range(c - block_w + 1, c + 1):
                    c1 = c0 + block_w - 1
                    if c0 < 1 or c1 > width - 2:
                        continue
                    rect = (r0, c0, r1, c1)
                    hit_indices = [j for j, (rr, cc, _dc) in enumerate(states) if _in_rect(rect, rr, cc)]
                    if not hit_indices or hit_indices[0] != idx:
                        continue
                    if any(states[j][2] != direction for j in hit_indices):
                        continue
                    area = block_h * block_w
                    if area == 1 and random.random() < 0.8:
                        continue
                    candidates.append(
                        {
                            "rect": rect,
                            "dir": int(direction),
                            "first_index": int(idx),
                            "area": int(area),
                        }
                    )

    random.shuffle(candidates)
    unique: list[dict[str, int | BBox]] = []
    seen: set[BBox] = set()
    for cand in candidates:
        rect = cand["rect"]
        assert isinstance(rect, tuple)
        if rect in seen:
            continue
        seen.add(rect)
        unique.append(cand)
    unique.sort(key=lambda cand: (int(cand["area"]) <= 1, -int(cand["area"]), int(cand["first_index"])))
    return unique


def _expected_output(input_grid: np.ndarray, rects: list[BBox]) -> np.ndarray:
    expected = np.asarray(input_grid, dtype=int).copy()
    states = _ray_trace(expected.shape, _find_blue_start(expected))
    for rect in rects:
        direction = _first_direction_in_rect(rect, states)
        if direction is None:
            continue
        color = RED if direction > 0 else TEAL
        r0, c0, r1, c1 = rect
        expected[r0 : r1 + 1, c0 : c1 + 1] = color
    return expected


def _validate_instance(input_grid: np.ndarray, rects: list[BBox]) -> bool:
    try:
        _find_blue_start(input_grid)
        grey_components = _components_mask(input_grid == GREY, connectivity=4)
        if len(grey_components) != len(rects):
            return False

        rect_set = set(rects)
        for coords in grey_components:
            bbox = _bbox_coords(coords)
            area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            if coords.shape[0] != area or bbox not in rect_set:
                return False

        out = main(input_grid)
        expected = _expected_output(input_grid, rects)
        if not np.array_equal(out, expected):
            return False

        return bool(np.any(out == RED) and np.any(out == TEAL) and np.any(out == GREY))
    except AssertionError:
        return False


def _generate_instance() -> tuple[np.ndarray, list[BBox]] | None:
    height = random.randint(MIN_H, MAX_H)
    width = random.randint(MIN_W, MAX_W)
    start_c = random.randint(1, width - 3)
    states = _ray_trace((height, width), start_c)
    if {dc for _r, _c, dc in states} != {1, -1}:
        return None

    red_candidates = _candidate_hit_rects(height, width, states, direction=1)
    teal_candidates = _candidate_hit_rects(height, width, states, direction=-1)
    if not red_candidates or not teal_candidates:
        return None

    chosen: list[dict[str, int | BBox]] = []
    red_pool = red_candidates[:40].copy()
    teal_pool = teal_candidates[:40].copy()
    random.shuffle(red_pool)
    random.shuffle(teal_pool)

    pair: tuple[dict[str, int | BBox], dict[str, int | BBox]] | None = None
    for red_cand in red_pool:
        for teal_cand in teal_pool:
            red_rect = red_cand["rect"]
            teal_rect = teal_cand["rect"]
            assert isinstance(red_rect, tuple) and isinstance(teal_rect, tuple)
            if not _rect_conflict(red_rect, teal_rect):
                pair = (red_cand, teal_cand)
                break
        if pair is not None:
            break
    if pair is None:
        return None

    chosen.extend(pair)
    extra_hit_target = random.randint(0, 2)
    extra_pool = (red_candidates[:50] + teal_candidates[:50]).copy()
    random.shuffle(extra_pool)
    for cand in extra_pool:
        if len(chosen) >= 2 + extra_hit_target:
            break
        rect = cand["rect"]
        assert isinstance(rect, tuple)
        if any(rect == other["rect"] for other in chosen):
            continue
        if any(_rect_conflict(rect, other["rect"]) for other in chosen):
            continue
        chosen.append(cand)

    rects = [cand["rect"] for cand in chosen]
    assert all(isinstance(rect, tuple) for rect in rects)
    rects = [rect for rect in rects if isinstance(rect, tuple)]

    target_unhit = random.randint(1, 3)
    attempts = 0
    while sum(_first_direction_in_rect(rect, states) is None for rect in rects) < target_unhit and attempts < PLACEMENT_ATTEMPTS:
        attempts += 1
        block_h = random.randint(1, 3)
        block_w = random.randint(1, 3)
        if block_h * block_w == 1 and random.random() < 0.7:
            continue

        r0 = random.randint(1, height - block_h - 2)
        c0 = random.randint(1, width - block_w - 2)
        rect = (r0, c0, r0 + block_h - 1, c0 + block_w - 1)
        if any(_rect_conflict(rect, other) for other in rects):
            continue
        if any(_in_rect(rect, r, c) for r, c, _dc in states):
            continue
        rects.append(rect)

    if not any(_first_direction_in_rect(rect, states) is None for rect in rects):
        return None

    input_grid = _build_input(height, width, start_c, rects)
    if not _validate_instance(input_grid, rects):
        return None
    return input_grid, rects


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        generated = _generate_instance()
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid reflected-ray instance")
