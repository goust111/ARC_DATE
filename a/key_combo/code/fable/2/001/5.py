from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox, bbox_coords, bbox_h, bbox_w, components_mask, crop_bbox, find_components, get_bg, paste


BACKGROUND = 0
GEN_ATTEMPTS = 400
def _full_grid_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _extract_notches(board: dict) -> dict[tuple[int, int], dict]:
    r0, c0, r1, c1 = map(int, board["bbox"])
    h, w = bbox_h(board["bbox"]), bbox_w(board["bbox"])
    solid = np.zeros((h, w), dtype=bool)
    rel = board["coords"] - np.array([r0, c0], dtype=int)
    solid[rel[:, 0], rel[:, 1]] = True

    notches: dict[tuple[int, int], dict] = {}
    for pts in components_mask(~solid, connectivity=4):
        if not np.any(pts[:, 0] == 0):
            continue
        local_bbox = bbox_coords(pts)
        nr0, nc0, nr1, nc1 = map(int, local_bbox)
        patch = np.zeros((nr1 - nr0 + 1, nc1 - nc0 + 1), dtype=bool)
        patch[pts[:, 0] - nr0, pts[:, 1] - nc0] = True
        global_bbox = (r0 + nr0, c0 + nc0, r0 + nr1, c0 + nc1)
        span = (global_bbox[1], global_bbox[3])
        notches[span] = {"bbox": global_bbox, "mask": patch}
    return notches


def _drop_until_contact(board_mask: np.ndarray, sprite_mask: np.ndarray, start_r: int, c0: int) -> int:
    h, w = sprite_mask.shape
    H, W = board_mask.shape
    r = int(start_r)
    c0 = int(c0)
    assert 0 <= c0 and c0 + w <= W
    assert 0 <= r and r + h <= H
    while r + 1 <= H - h:
        if np.any(board_mask[r + 1 : r + 1 + h, c0 : c0 + w] & sprite_mask):
            break
        r += 1
    return r


def _same_shape(a: np.ndarray, b: np.ndarray | None) -> bool:
    return b is not None and a.shape == b.shape and np.array_equal(a, b)


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, dict], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    boards = [comp for comp in comps if bbox_w(comp["bbox"]) == g.shape[1]]
    boards.sort(key=lambda comp: comp["bbox"][0])
    assert len(boards) == 2, "expected exactly two full-width boards"

    board_ids = {id(board) for board in boards}
    top_limit = int(boards[0]["bbox"][0])
    sprites = [comp for comp in comps if id(comp) not in board_ids and int(comp["bbox"][2]) < top_limit]
    sprites.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    assert sprites, "expected at least one sprite"

    notch_maps: dict[int, dict] = {}
    board_masks: dict[int, np.ndarray] = {}
    for board in boards:
        notch_maps[id(board)] = _extract_notches(board)
        board_masks[id(board)] = _full_grid_mask(g.shape, board["coords"])
    return boards, sprites, notch_maps, board_masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each sprite through two horizontal sieve boards, passing a board only when its aligned notch is exactly congruent.

    Input structure:
    - Black is the background.
    - Two monochrome horizontal boards span the full width, each with one or more background notches cut into its top edge.
    - Several smaller monochrome sprites float above the upper board; their column ranges are pairwise disjoint.

    Rule:
    - Each sprite falls straight down within its own column span.
    - At a board, compare the sprite's binary shape with the notch directly below the same column span.
    - If they are exactly the same shape, the sprite passes that board and continues falling below it; otherwise it stops at the last non-overlapping row that first contacts the board outline.
    - A sprite that passes both boards keeps falling until its bottom touches the grid bottom.

    Key insight:
    - The two boards form a cascade of shape-equality tests: every sprite's final resting place is determined by which aligned notch tests it passes in order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    boards, sprites, notch_maps, board_masks = _analyze_scene(g)
    top_board, bottom_board = boards

    out = g.copy()
    for sprite in sprites:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = bg

    for sprite in sprites:
        sr0, sc0, sr1, sc1 = map(int, sprite["bbox"])
        sprite_patch = crop_bbox(g, sprite["bbox"])
        sprite_mask = sprite_patch != bg
        h = bbox_h(sprite["bbox"])
        span = (sc0, sc1)

        top_notch = notch_maps[id(top_board)].get(span)
        passes_top = _same_shape(sprite_mask, None if top_notch is None else top_notch["mask"])
        if not passes_top:
            final_r = _drop_until_contact(board_masks[id(top_board)], sprite_mask, sr0, sc0)
        else:
            below_top = int(top_board["bbox"][2]) + 1
            bottom_notch = notch_maps[id(bottom_board)].get(span)
            passes_bottom = _same_shape(sprite_mask, None if bottom_notch is None else bottom_notch["mask"])
            if not passes_bottom:
                final_r = _drop_until_contact(board_masks[id(bottom_board)], sprite_mask, below_top, sc0)
            else:
                final_r = g.shape[0] - h

        target = out[final_r : final_r + sprite_patch.shape[0], sc0 : sc0 + sprite_patch.shape[1]]
        assert np.all(target[sprite_mask] == bg), "sprite placement collided unexpectedly"
        paste(out, sprite_patch, final_r, sc0, transparent=bg)

    return out


def _random_shape(height: int, width: int) -> np.ndarray:
    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        depths = [random.randint(1, height) for _ in range(width)]
        if max(depths) != height:
            continue
        if all(depth == height for depth in depths):
            continue
        for c, depth in enumerate(depths):
            mask[:depth, c] = True
        if int(mask.sum()) < 3:
            continue
        return mask
    raise RuntimeError("failed to sample shape")


def _different_shape(reference: np.ndarray) -> np.ndarray:
    width = reference.shape[1]
    for _ in range(200):
        height = random.randint(2, 4)
        candidate = _random_shape(height, width)
        if candidate.shape != reference.shape or not np.array_equal(candidate, reference):
            return candidate
    raise RuntimeError("failed to sample a different shape")


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _expected_row(spec: dict, top_board_mask: np.ndarray, bottom_board_mask: np.ndarray, top_board_bottom: int, H: int) -> tuple[int, str]:
    sprite_mask = spec["sprite"]
    start_r = spec["start_r"]
    c0 = spec["c0"]
    if not _same_shape(sprite_mask, spec["top_notch"]):
        return _drop_until_contact(top_board_mask, sprite_mask, start_r, c0), "top"
    lower_start = top_board_bottom + 1
    if not _same_shape(sprite_mask, spec["bottom_notch"]):
        return _drop_until_contact(bottom_board_mask, sprite_mask, lower_start, c0), "middle"
    return H - sprite_mask.shape[0], "bottom"


def _gen_attempt() -> np.ndarray | None:
    board_color = random.randint(1, 9)
    n_lanes = random.randint(3, 5)
    base_outcomes = ["top", "middle", "bottom"]
    if n_lanes > 3:
        base_outcomes.extend(random.choices(["top", "middle", "bottom"], k=n_lanes - 3))
    random.shuffle(base_outcomes)

    top_failure_indices = [i for i, outcome in enumerate(base_outcomes) if outcome == "top"]
    blank_top_index = random.choice(top_failure_indices) if top_failure_indices and random.random() < 0.6 else None

    sprite_palette = [color for color in range(1, 10) if color != board_color]
    if len(sprite_palette) < n_lanes:
        return None
    sprite_colors = random.sample(sprite_palette, k=n_lanes)

    lane_specs: list[dict] = []
    for idx, outcome in enumerate(base_outcomes):
        sprite = _random_shape(random.randint(2, 4), random.randint(2, 4))
        if outcome == "top" and idx == blank_top_index:
            top_notch = None
            bottom_notch = None
        elif outcome == "top":
            top_notch = _different_shape(sprite)
            bottom_notch = _different_shape(sprite)
        elif outcome == "middle":
            top_notch = sprite.copy()
            bottom_notch = _different_shape(sprite)
        else:
            top_notch = sprite.copy()
            bottom_notch = sprite.copy()
        lane_specs.append(
            {
                "color": sprite_colors[idx],
                "sprite": sprite,
                "top_notch": top_notch,
                "bottom_notch": bottom_notch,
                "outcome": outcome,
                "width": sprite.shape[1],
            }
        )

    widths = [spec["width"] for spec in lane_specs]
    gaps = [random.randint(2, 4) for _ in range(n_lanes - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    min_width = left_margin + right_margin + sum(widths) + sum(gaps)
    if min_width > 30:
        return None
    width = max(18, min_width)
    width += random.randint(0, min(30 - width, 4))
    slack = width - min_width
    buckets = [left_margin, *gaps, right_margin]
    for _ in range(slack):
        buckets[random.randrange(len(buckets))] += 1
    left_margin = buckets[0]
    gaps = buckets[1:-1]

    top_notch_max = max((spec["top_notch"].shape[0] for spec in lane_specs if spec["top_notch"] is not None), default=1)
    bottom_notch_max = max((spec["bottom_notch"].shape[0] for spec in lane_specs if spec["bottom_notch"] is not None), default=1)
    max_sprite_h = max(spec["sprite"].shape[0] for spec in lane_specs)
    top_thickness = max(4, top_notch_max + random.randint(1, 2))
    bottom_thickness = max(4, bottom_notch_max + random.randint(1, 2))
    top_gap = max_sprite_h + random.randint(2, 4)
    between_gap = max_sprite_h + random.randint(3, 5)
    bottom_gap = max_sprite_h + random.randint(2, 4)

    top_board_r = top_gap
    bottom_board_r = top_board_r + top_thickness + between_gap
    height = bottom_board_r + bottom_thickness + bottom_gap
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[top_board_r : top_board_r + top_thickness, :] = board_color
    grid[bottom_board_r : bottom_board_r + bottom_thickness, :] = board_color

    c = left_margin
    for idx, spec in enumerate(lane_specs):
        sprite = spec["sprite"]
        h, w = sprite.shape
        spec["c0"] = c
        spec["c1"] = c + w - 1
        spec["start_r"] = random.randint(0, top_board_r - h - 2)
        paste(grid, _mask_to_patch(sprite, spec["color"]), spec["start_r"], c, transparent=BACKGROUND)

        if spec["top_notch"] is not None:
            notch = spec["top_notch"]
            board_patch = grid[top_board_r : top_board_r + notch.shape[0], c : c + w]
            board_patch[notch] = BACKGROUND
        if spec["bottom_notch"] is not None:
            notch = spec["bottom_notch"]
            board_patch = grid[bottom_board_r : bottom_board_r + notch.shape[0], c : c + w]
            board_patch[notch] = BACKGROUND

        if idx < len(gaps):
            c += w + gaps[idx]

    out = main(grid)
    top_board_mask = np.zeros_like(grid, dtype=bool)
    top_board_mask[top_board_r : top_board_r + top_thickness, :] = True
    if any(spec["top_notch"] is not None for spec in lane_specs):
        for spec in lane_specs:
            if spec["top_notch"] is None:
                continue
            notch = spec["top_notch"]
            top_board_mask[top_board_r : top_board_r + notch.shape[0], spec["c0"] : spec["c1"] + 1][notch] = False

    bottom_board_mask = np.zeros_like(grid, dtype=bool)
    bottom_board_mask[bottom_board_r : bottom_board_r + bottom_thickness, :] = True
    if any(spec["bottom_notch"] is not None for spec in lane_specs):
        for spec in lane_specs:
            if spec["bottom_notch"] is None:
                continue
            notch = spec["bottom_notch"]
            bottom_board_mask[bottom_board_r : bottom_board_r + notch.shape[0], spec["c0"] : spec["c1"] + 1][notch] = False

    seen_outcomes = set()
    for spec in lane_specs:
        coords = np.argwhere(out == spec["color"])
        if coords.size == 0:
            return None
        expected_r, expected_kind = _expected_row(spec, top_board_mask, bottom_board_mask, top_board_r + top_thickness - 1, height)
        actual_bbox = bbox(out == spec["color"])
        if actual_bbox is None:
            return None
        expected_bbox = (expected_r, spec["c0"], expected_r + spec["sprite"].shape[0] - 1, spec["c1"])
        if actual_bbox != expected_bbox:
            return None
        seen_outcomes.add(expected_kind)

    if seen_outcomes != {"top", "middle", "bottom"}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-sieve input")
