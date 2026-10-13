import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg, in_bounds, paste


BG = 0
BLUE = 1
ORANGE = 7
SAFE_COLORS = [2, 3, 4, 5, 6, 8, 9]
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (20, 30)
PAIR_COUNT_CHOICES = [2, 3]
STOP_K_RANGE = (3, 5)
TARGET_SIZE_RANGE = (2, 4)
BAR_LENGTH_RANGE = (3, 6)
BAR_RADIUS_CHOICES = [1, 2]
GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 200
BAR_PLACEMENT_ATTEMPTS = 250


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sprite_patch(body_color: int, center_color: int) -> np.ndarray:
    patch = np.full((3, 3), int(body_color), dtype=int)
    patch[1, 1] = int(center_color)
    return patch


def _patch_bbox(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return int(r - 1), int(c - 1), int(r + 1), int(c + 1)


def _patch_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _obstacle_forbids_cell(obstacle: dict, cell: tuple[int, int]) -> bool:
    rr, cc = cell
    radius = int(obstacle["radius"])
    for r, c in obstacle["coords"]:
        if abs(int(r) - rr) + abs(int(c) - cc) <= radius:
            return True
    return False


def _patch_hits_obstacles(center: tuple[int, int], obstacles: list[dict]) -> bool:
    for cell in _patch_cells(center):
        for obstacle in obstacles:
            if _obstacle_forbids_cell(obstacle, cell):
                return True
    return False


def _build_forbidden(shape: tuple[int, int], obstacles: list[dict]) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for obstacle in obstacles:
        rows = [r for r, _ in obstacle["coords"]]
        cols = [c for _, c in obstacle["coords"]]
        radius = int(obstacle["radius"])
        r0 = max(0, min(rows) - radius)
        r1 = min(h - 1, max(rows) + radius)
        c0 = max(0, min(cols) - radius)
        c1 = min(w - 1, max(cols) + radius)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if _obstacle_forbids_cell(obstacle, (r, c)):
                    forbidden[r, c] = True
    return forbidden


def _find_sprites(grid: np.ndarray) -> list[dict]:
    sprites = []
    for coords in components_mask(grid != BG, connectivity=4):
        if coords.shape[0] != 9:
            continue
        box = _bbox(coords)
        if box[2] - box[0] != 2 or box[3] - box[1] != 2:
            continue
        r0, c0, r1, c1 = box
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BG):
            continue
        border = np.array(
            [
                patch[0, 0],
                patch[0, 1],
                patch[0, 2],
                patch[1, 0],
                patch[1, 2],
                patch[2, 0],
                patch[2, 1],
                patch[2, 2],
            ]
        )
        if len(set(map(int, border))) != 1:
            continue
        body_color = int(border[0])
        center_color = int(patch[1, 1])
        if center_color == body_color:
            continue
        sprites.append(
            {
                "bbox": box,
                "center": (int(r0 + 1), int(c0 + 1)),
                "body_color": body_color,
                "center_color": center_color,
                "patch": patch.copy(),
            }
        )
    sprites.sort(key=lambda item: item["center"])
    return sprites


def _find_targets(grid: np.ndarray, center_colors: set[int]) -> dict[int, dict]:
    targets: dict[int, dict] = {}
    for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color not in center_colors:
            continue
        box = tuple(map(int, comp["bbox"]))
        if comp["area"] < 4 or _bbox_area(box) != int(comp["area"]):
            continue
        assert color not in targets, "each target color must map to exactly one solid rectangle"
        targets[color] = {"color": color, "bbox": box}
    assert len(targets) == len(center_colors), "each sprite center color must have one solid target rectangle"
    return targets


def _find_obstacles(grid: np.ndarray) -> list[dict]:
    mask = (grid == BLUE) | (grid == ORANGE)
    obstacles = []
    for coords in components_mask(mask, connectivity=4):
        box = _bbox(coords)
        h = int(box[2] - box[0] + 1)
        w = int(box[3] - box[1] + 1)
        length = max(h, w)
        colors = {int(grid[r, c]) for r, c in coords}
        assert colors <= {BLUE, ORANGE}
        assert min(h, w) == 1 and coords.shape[0] == length and 3 <= length <= 6
        blue_count = sum(1 for r, c in coords if int(grid[r, c]) == BLUE)
        assert BLUE in colors and ORANGE in colors and 1 <= blue_count < length
        ordered = sorted((int(r), int(c)) for r, c in coords)
        obstacles.append(
            {
                "bbox": box,
                "coords": ordered,
                "radius": int(blue_count),
            }
        )
    obstacles.sort(key=lambda item: item["bbox"])
    return obstacles


def _pair_sprite_to_target(sprite: dict, target: dict) -> dict:
    sr, sc = sprite["center"]
    r0, c0, r1, c1 = target["bbox"]
    horizontal = r0 <= sr <= r1 and (sc < c0 or sc > c1)
    vertical = c0 <= sc <= c1 and (sr < r0 or sr > r1)
    assert horizontal ^ vertical, "sprite and target must be uniquely aligned"
    if horizontal:
        dr, dc = 0, (1 if sc < c0 else -1)
    else:
        dr, dc = (1 if sr < r0 else -1), 0
    pair = dict(sprite)
    pair["target_bbox"] = tuple(map(int, target["bbox"]))
    pair["dir"] = (int(dr), int(dc))
    return pair


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG, "background must be black"
    sprites = _find_sprites(g)
    assert 2 <= len(sprites) <= 3, "expected 2-3 sprites"
    center_colors = {int(sprite["center_color"]) for sprite in sprites}
    assert len(center_colors) == len(sprites), "sprite center colors must be unique"
    targets = _find_targets(g, center_colors)
    obstacles = _find_obstacles(g)
    assert 2 <= len(obstacles) <= 4, "expected 2-4 obstacle bars"
    pairs = [_pair_sprite_to_target(sprite, targets[int(sprite["center_color"])]) for sprite in sprites]
    return {"grid": g, "sprites": sprites, "pairs": pairs, "obstacles": obstacles}


def _simulate_pair(pair: dict, obstacles: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    results = []
    dr, dc = pair["dir"]
    for k in range(1, 15):
        center = (int(pair["center"][0] + 3 * dr * k), int(pair["center"][1] + 3 * dc * k))
        box = _patch_bbox(center)
        if not (in_bounds(shape, box[0], box[1]) and in_bounds(shape, box[2], box[3])):
            break
        safe = not _patch_hits_obstacles(center, obstacles)
        overlaps = _bbox_intersects(box, pair["target_bbox"])
        results.append(
            {
                "k": int(k),
                "center": center,
                "bbox": box,
                "safe": bool(safe),
                "overlaps_target": bool(overlaps),
            }
        )
        if safe and overlaps:
            break
    assert results, "each pair must have at least one candidate landing"
    return results


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each 3x3 ring sprite toward its same-colored target rectangle, skipping candidate stamps whose 3x3 landing zone enters any obstacle keep-out region.

    Input structure:
    - The background is black.
    - There are 2-3 filled 3x3 sprites whose eight border pixels share one body color and whose center pixel has a unique target color.
    - For each target color there is exactly one solid same-colored rectangle, aligned with its sprite in the same row band or column band.
    - There are 2-4 one-pixel-wide blue/orange obstacle bars; each bar is straight, length 3-6, and its blue-pixel count encodes that bar's Manhattan keep-out radius.

    Rule:
    - Pair every sprite with the unique solid rectangle whose color matches the sprite center.
    - From the sprite center, march toward the rectangle in the aligned cardinal direction in steps of 3 cells, producing candidate centers for copied 3x3 stamps.
    - A candidate is forbidden if any cell of its 3x3 stamp lies within Manhattan distance <= r of any pixel of any obstacle bar, where r is that bar's blue count.
    - Place only the safe sprite copies; skip forbidden candidates; stop once a placed copy overlaps the paired rectangle.
    - Keep the original sprites and rectangles, let placed copies overwrite overlapped target cells, and recolor every blue obstacle pixel to orange in the final output.

    Key insight:
    - The blue pixels on each obstacle bar are not decorative: they encode a Manhattan keep-out radius that deletes specific 3-step stamp landings, so the repeated pattern is understood by combining periodic copying with obstacle-based skipping.
    """
    scene = _analyze_input(np.asarray(grid, dtype=int))
    out = scene["grid"].copy()
    forbidden = _build_forbidden(out.shape, scene["obstacles"])
    for pair in scene["pairs"]:
        placed_overlap = False
        dr, dc = pair["dir"]
        for k in range(1, 15):
            center = (int(pair["center"][0] + 3 * dr * k), int(pair["center"][1] + 3 * dc * k))
            r0, c0, r1, c1 = _patch_bbox(center)
            if not (in_bounds(out.shape, r0, c0) and in_bounds(out.shape, r1, c1)):
                break
            if forbidden[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            paste(out, pair["patch"], r0, c0)
            if _bbox_intersects((r0, c0, r1, c1), pair["target_bbox"]):
                placed_overlap = True
                break
        assert placed_overlap, "each sprite path must end by overlapping its target rectangle"
    out[out == BLUE] = ORANGE
    return out


def _sample_pair_layout(
    rng: random.Random,
    h: int,
    w: int,
    body_color: int,
    center_color: int,
    reserved: np.ndarray,
) -> dict | None:
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        stop_k = rng.randint(STOP_K_RANGE[0], STOP_K_RANGE[1])
        th = rng.randint(TARGET_SIZE_RANGE[0], TARGET_SIZE_RANGE[1])
        tw = rng.randint(TARGET_SIZE_RANGE[0], TARGET_SIZE_RANGE[1])
        orient = rng.choice(["h", "v"])
        sign = rng.choice([-1, 1])

        if orient == "h":
            sr = rng.randint(2, h - 3)
            tr0_lo = max(1, sr - th + 1)
            tr0_hi = min(sr, h - th - 1)
            if tr0_lo > tr0_hi:
                continue
            tr0 = rng.randint(tr0_lo, tr0_hi)
            if sign > 0:
                sc_hi = w - tw - 3 * stop_k
                if sc_hi < 1:
                    continue
                sc = rng.randint(1, sc_hi)
                target_c0 = sc + 3 * stop_k - 1
                path_bbox = (int(sr - 1), int(sc + 2), int(sr + 1), int(sc + 3 * stop_k + 1))
            else:
                sc_lo = 3 * stop_k + tw - 1
                if sc_lo > w - 2:
                    continue
                sc = rng.randint(sc_lo, w - 2)
                target_c0 = sc - 3 * stop_k - tw + 2
                path_bbox = (int(sr - 1), int(sc - 3 * stop_k - 1), int(sr + 1), int(sc - 2))
            target_bbox = (int(tr0), int(target_c0), int(tr0 + th - 1), int(target_c0 + tw - 1))
            direction = (0, int(sign))
        else:
            sc = rng.randint(2, w - 3)
            tc0_lo = max(1, sc - tw + 1)
            tc0_hi = min(sc, w - tw - 1)
            if tc0_lo > tc0_hi:
                continue
            tc0 = rng.randint(tc0_lo, tc0_hi)
            if sign > 0:
                sr_hi = h - th - 3 * stop_k
                if sr_hi < 1:
                    continue
                sr = rng.randint(1, sr_hi)
                target_r0 = sr + 3 * stop_k - 1
                path_bbox = (int(sr + 2), int(sc - 1), int(sr + 3 * stop_k + 1), int(sc + 1))
            else:
                sr_lo = 3 * stop_k + th - 1
                if sr_lo > h - 2:
                    continue
                sr = rng.randint(sr_lo, h - 2)
                target_r0 = sr - 3 * stop_k - th + 2
                path_bbox = (int(sr - 3 * stop_k - 1), int(sc - 1), int(sr - 2), int(sc + 1))
            target_bbox = (int(target_r0), int(tc0), int(target_r0 + th - 1), int(tc0 + tw - 1))
            direction = (int(sign), 0)

        sprite_bbox = (int(sr - 1), int(sc - 1), int(sr + 1), int(sc + 1))
        reserve = np.zeros_like(reserved, dtype=bool)
        reserve[sprite_bbox[0] : sprite_bbox[2] + 1, sprite_bbox[1] : sprite_bbox[3] + 1] = True
        reserve[path_bbox[0] : path_bbox[2] + 1, path_bbox[1] : path_bbox[3] + 1] = True
        reserve[target_bbox[0] : target_bbox[2] + 1, target_bbox[1] : target_bbox[3] + 1] = True
        if np.any(reserve & reserved):
            continue

        return {
            "bbox": sprite_bbox,
            "center": (int(sr), int(sc)),
            "body_color": int(body_color),
            "center_color": int(center_color),
            "patch": _sprite_patch(body_color, center_color),
            "target_bbox": target_bbox,
            "dir": direction,
            "stop_k": int(stop_k),
            "path_bbox": path_bbox,
        }
    return None


def _occupied_object_mask(shape: tuple[int, int], pairs: list[dict], obstacles: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for pair in pairs:
        r0, c0, r1, c1 = pair["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        tr0, tc0, tr1, tc1 = pair["target_bbox"]
        mask[tr0 : tr1 + 1, tc0 : tc1 + 1] = True
    for obstacle in obstacles:
        for r, c in obstacle["coords"]:
            mask[r, c] = True
    return mask


def _affected_candidates(pair: dict, obstacles: list[dict]) -> set[int]:
    hits = set()
    for k in range(1, int(pair["stop_k"]) + 1):
        center = (
            int(pair["center"][0] + 3 * pair["dir"][0] * k),
            int(pair["center"][1] + 3 * pair["dir"][1] * k),
        )
        if _patch_hits_obstacles(center, obstacles):
            hits.add(int(k))
    return hits


def _sample_obstacle_for_pair(
    rng: random.Random,
    shape: tuple[int, int],
    pair: dict,
    all_pairs: list[dict],
    existing_obstacles: list[dict],
) -> dict | None:
    h, w = shape
    occupied = _occupied_object_mask(shape, all_pairs, existing_obstacles)
    before_self = _affected_candidates(pair, existing_obstacles)
    candidate_ks = [k for k in range(1, int(pair["stop_k"])) if k not in before_self]
    if not candidate_ks:
        return None

    for _ in range(BAR_PLACEMENT_ATTEMPTS):
        k_block = rng.choice(candidate_ks)
        center = (
            int(pair["center"][0] + 3 * pair["dir"][0] * k_block),
            int(pair["center"][1] + 3 * pair["dir"][1] * k_block),
        )
        pr0, pc0, pr1, pc1 = _patch_bbox(center)
        length = rng.randint(BAR_LENGTH_RANGE[0], BAR_LENGTH_RANGE[1])
        radius = rng.choice([r for r in BAR_RADIUS_CHOICES if r < length])
        orient = rng.choice(["h", "v"])

        if orient == "h":
            row_lo = max(0, pr0 - radius)
            row_hi = min(h - 1, pr1 + radius)
            col_lo = max(0, pc0 - length + 1)
            col_hi = min(pc1, w - length)
            if row_lo > row_hi or col_lo > col_hi:
                continue
            rr = rng.randint(row_lo, row_hi)
            cc = rng.randint(col_lo, col_hi)
            coords = [(int(rr), int(cc + i)) for i in range(length)]
        else:
            row_lo = max(0, pr0 - length + 1)
            row_hi = min(pr1, h - length)
            col_lo = max(0, pc0 - radius)
            col_hi = min(w - 1, pc1 + radius)
            if row_lo > row_hi or col_lo > col_hi:
                continue
            rr = rng.randint(row_lo, row_hi)
            cc = rng.randint(col_lo, col_hi)
            coords = [(int(rr + i), int(cc)) for i in range(length)]

        if any(occupied[r, c] for r, c in coords):
            continue

        blue_idx = set(rng.sample(range(length), radius))
        obstacle = {
            "bbox": (min(r for r, _ in coords), min(c for _, c in coords), max(r for r, _ in coords), max(c for _, c in coords)),
            "coords": coords,
            "radius": int(radius),
            "blue_idx": blue_idx,
        }
        updated = existing_obstacles + [obstacle]
        after_self = _affected_candidates(pair, updated)
        if int(pair["stop_k"]) in after_self:
            continue
        if k_block not in after_self - before_self:
            continue
        if len(after_self) >= int(pair["stop_k"]):
            continue
        if not any(k not in after_self for k in range(1, int(pair["stop_k"]))):
            continue

        cross_talk = False
        for other in all_pairs:
            if other is pair:
                continue
            if _affected_candidates(other, updated) != _affected_candidates(other, existing_obstacles):
                cross_talk = True
                break
        if cross_talk:
            continue
        return obstacle
    return None


def _render_scene(shape: tuple[int, int], pairs: list[dict], obstacles: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for pair in pairs:
        r0, c0, _, _ = pair["bbox"]
        paste(grid, pair["patch"], r0, c0)
        tr0, tc0, tr1, tc1 = pair["target_bbox"]
        grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = int(pair["center_color"])
    for obstacle in obstacles:
        for idx, (r, c) in enumerate(obstacle["coords"]):
            grid[r, c] = BLUE if idx in obstacle["blue_idx"] else ORANGE
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        scene = _analyze_input(grid)
        all_path_boxes: list[tuple[int, int, int, int]] = []
        for pair in scene["pairs"]:
            sims = _simulate_pair(pair, scene["obstacles"], grid.shape)
            stop_hits = [item for item in sims if item["safe"] and item["overlaps_target"]]
            assert stop_hits, "each path needs a safe overlapping terminal stamp"
            stop_k = int(stop_hits[0]["k"])
            assert any((not item["safe"]) and item["k"] < stop_k for item in sims), "each path must include a skipped candidate"
            assert any(item["safe"] and item["k"] < stop_k for item in sims), "each path must also include an earlier placed candidate"
            for item in sims:
                if item["k"] <= stop_k:
                    all_path_boxes.append(item["bbox"])

        for i in range(len(all_path_boxes)):
            for j in range(i + 1, len(all_path_boxes)):
                a = all_path_boxes[i]
                b = all_path_boxes[j]
                if _bbox_intersects(a, b):
                    a_pair = (a[0] + a[2], a[1] + a[3])
                    b_pair = (b[0] + b[2], b[1] + b[3])
                    assert a_pair == b_pair, "different sprite paths must stay disjoint"

        out = main(grid)
        assert not np.any(out == BLUE), "blue obstacle code must disappear in output"
        for pair in scene["pairs"]:
            sims = _simulate_pair(pair, scene["obstacles"], grid.shape)
            for item in sims:
                if item["safe"]:
                    r0, c0, r1, c1 = item["bbox"]
                    if _bbox_intersects(item["bbox"], pair["target_bbox"]) or item["k"] < sims[-1]["k"]:
                        assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], pair["patch"])
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1] - 1)
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1] - 1)
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    colors = rng.sample(SAFE_COLORS, 2 * pair_count)
    reserved = np.zeros((h, w), dtype=bool)
    pairs = []

    for idx in range(pair_count):
        body_color = colors[2 * idx]
        center_color = colors[2 * idx + 1]
        pair = _sample_pair_layout(rng, h, w, body_color, center_color, reserved)
        if pair is None:
            return None
        pairs.append(pair)
        r0, c0, r1, c1 = pair["bbox"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        pr0, pc0, pr1, pc1 = pair["path_bbox"]
        reserved[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        tr0, tc0, tr1, tc1 = pair["target_bbox"]
        reserved[tr0 : tr1 + 1, tc0 : tc1 + 1] = True

    obstacles: list[dict] = []
    for pair in pairs:
        obstacle = _sample_obstacle_for_pair(rng, (h, w), pair, pairs, obstacles)
        if obstacle is None:
            return None
        obstacles.append(obstacle)

    if len(obstacles) < 4 and rng.random() < 0.55:
        extra_pair_order = pairs[:]
        rng.shuffle(extra_pair_order)
        added = False
        for pair in extra_pair_order:
            obstacle = _sample_obstacle_for_pair(rng, (h, w), pair, pairs, obstacles)
            if obstacle is not None:
                obstacles.append(obstacle)
                added = True
                break
        if not added and len(pairs) == 2 and rng.random() < 0.35:
            return None

    grid = _render_scene((h, w), pairs, obstacles)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
