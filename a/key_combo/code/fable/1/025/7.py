from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, get_bg


MIRROR_COLOR = Color.GREY
MAX_GENERATION_ATTEMPTS = 500

BG_CHOICES = [c for c in Color.ALL_COLORS if c != MIRROR_COLOR]
SPRITE_COLOR_POOL = [c for c in Color.NOT_BLACK if c != MIRROR_COLOR]


def _mirror_from_coords(coords: np.ndarray, mirror_id: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    area = int(len(coords))
    if c0 == c1:
        assert area == r1 - r0 + 1, "vertical mirror must be contiguous"
        return {
            "id": mirror_id,
            "orientation": "v",
            "axis": int(c0),
            "span0": int(r0),
            "span1": int(r1),
            "length": int(r1 - r0 + 1),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
        }
    if r0 == r1:
        assert area == c1 - c0 + 1, "horizontal mirror must be contiguous"
        return {
            "id": mirror_id,
            "orientation": "h",
            "axis": int(r0),
            "span0": int(c0),
            "span1": int(c1),
            "length": int(c1 - c0 + 1),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
        }
    raise AssertionError("mirror components must be axis-aligned straight bars")


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    mirror_mask = g == MIRROR_COLOR
    assert np.any(mirror_mask), "expected at least one grey mirror"
    mirrors = [
        _mirror_from_coords(coords, mirror_id)
        for mirror_id, coords in enumerate(components_mask(mirror_mask, connectivity=4))
    ]
    assert len(mirrors) >= 2, "expected several mirror bars"

    sprite_mask = (g != bg) & ~mirror_mask
    sprite_components = components_mask(sprite_mask, connectivity=4)
    assert len(sprite_components) == 1, "expected one connected multicolor sprite"
    sprite_coords = sprite_components[0]
    sprite_bbox = bbox_coords(sprite_coords)
    sprite_crop = crop_bbox(g, sprite_bbox)
    sprite_colors = sorted({int(g[r, c]) for r, c in sprite_coords})
    assert len(sprite_colors) >= 2, "sprite should be multicolor"

    return {
        "grid": g,
        "bg": bg,
        "mirrors": mirrors,
        "mirror_mask": mirror_mask,
        "sprite_mask": sprite_mask,
        "sprite_coords": [tuple(map(int, rc)) for rc in sprite_coords],
        "sprite_bbox": sprite_bbox,
        "sprite_crop": sprite_crop,
    }


def _candidate_mirrors(pos: tuple[int, int], mirrors: list[dict], used: set[int]) -> list[tuple[int, dict]]:
    r, c = pos
    candidates: list[tuple[int, dict]] = []
    for mirror in mirrors:
        if mirror["id"] in used:
            continue
        if mirror["orientation"] == "v":
            if mirror["span0"] <= r <= mirror["span1"] and c != mirror["axis"]:
                candidates.append((abs(c - mirror["axis"]), mirror))
        else:
            if mirror["span0"] <= c <= mirror["span1"] and r != mirror["axis"]:
                candidates.append((abs(r - mirror["axis"]), mirror))
    return candidates


def _reflect(pos: tuple[int, int], mirror: dict) -> tuple[int, int]:
    r, c = pos
    axis = int(mirror["axis"])
    if mirror["orientation"] == "v":
        return int(r), int(2 * axis - c)
    return int(2 * axis - r), int(c)


def _trace_pixel(
    start: tuple[int, int], mirrors: list[dict], shape: tuple[int, int]
) -> tuple[tuple[int, int], list[tuple[int, int]], list[int]]:
    h, w = map(int, shape)
    current = tuple(map(int, start))
    stops: list[tuple[int, int]] = []
    mirror_ids: list[int] = []
    used: set[int] = set()

    while True:
        candidates = _candidate_mirrors(current, mirrors, used)
        if not candidates:
            return current, stops, mirror_ids

        best_dist = min(distance for distance, _mirror in candidates)
        best = [mirror for distance, mirror in candidates if distance == best_dist]
        assert len(best) == 1, "each relay step must have a unique nearest mirror"

        mirror = best[0]
        current = _reflect(current, mirror)
        assert 0 <= current[0] < h and 0 <= current[1] < w, "reflections must stay in bounds"
        used.add(int(mirror["id"]))
        mirror_ids.append(int(mirror["id"]))
        stops.append(current)


def _trace_scene(grid: np.ndarray) -> tuple[dict, list[dict]]:
    scene = _parse_scene(grid)
    traces: list[dict] = []
    for pos in scene["sprite_coords"]:
        color = int(scene["grid"][pos])
        final, stops, mirror_ids = _trace_pixel(pos, scene["mirrors"], scene["grid"].shape)
        traces.append(
            {
                "start": pos,
                "color": color,
                "final": final,
                "stops": stops,
                "mirror_ids": mirror_ids,
            }
        )
    return scene, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay the multicolor sprite through the grey mirror bars and keep only each pixel's final reflected landing.

    Input structure:
    - A dominant background surrounds one connected multicolor sprite near the middle of the grid.
    - Several separate grey straight bars act as mirrors; each bar is axis-aligned, one cell thick, and offset from the sprite by background cells.
    - Different mirror spans cover different row or column bands, so a reflected band can land inside another mirror's span.

    Rule:
    - Treat each non-grey sprite pixel independently.
    - Repeatedly choose the unique nearest unused grey bar whose span covers that pixel's orthogonal projection, then reflect the pixel across that bar's line.
    - Stop when no unused covering mirror remains; draw only that final landing pixel in its original color, while keeping the original sprite and all mirror bars unchanged.

    Key insight:
    - The output is not the union of independent one-step mirror copies: some pixels are moved again by later mirrors, so only the last image along each relay path survives.
    """
    scene, traces = _trace_scene(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    out = g.copy()
    landings: dict[tuple[int, int], int] = {}

    for trace in traces:
        start = trace["start"]
        final = trace["final"]
        if final == start:
            continue
        fr, fc = final
        assert int(g[fr, fc]) == bg, "final reflected pixels must land on background"
        assert final not in landings, "different source pixels may not collapse onto one landing cell"
        landings[final] = int(trace["color"])

    for (r, c), color in landings.items():
        out[r, c] = int(color)
    return out


def _connected_mask(mask: np.ndarray) -> bool:
    return bool(np.any(mask)) and len(components_mask(mask, connectivity=4)) == 1


def _random_sprite_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    removable = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
        (0, 1),
        (1, 0),
        (h - 1, w - 2),
        (h - 2, w - 1),
    ]
    random.shuffle(removable)
    budget = random.randint(1, min(3, len(removable)))
    for r, c in removable:
        if budget == 0:
            break
        trial = mask.copy()
        trial[r, c] = False
        if _connected_mask(trial) and np.all(trial.any(axis=0)) and np.all(trial.any(axis=1)):
            mask = trial
            budget -= 1
    return mask


def _random_sprite_array(mask: np.ndarray, bg: int) -> np.ndarray:
    h, w = mask.shape
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    palette_pool = [color for color in SPRITE_COLOR_POOL if color != int(bg)]
    assert len(palette_pool) >= 2
    for _ in range(200):
        palette_size = random.randint(2, min(4, len(palette_pool)))
        palette = random.sample(palette_pool, palette_size)
        sprite = np.full((h, w), bg, dtype=int)
        for r, c in cells:
            sprite[r, c] = int(random.choice(palette))

        colored = {int(sprite[r, c]) for r, c in cells}
        if len(colored) < 2:
            continue
        if np.array_equal(sprite, np.fliplr(sprite)):
            continue
        if np.array_equal(sprite, np.flipud(sprite)):
            continue
        if np.array_equal(sprite, np.rot90(sprite, 2)):
            continue
        return sprite
    raise RuntimeError("failed to build an asymmetric sprite")


def _apply_dihedral(grid: np.ndarray, op: int) -> np.ndarray:
    op = int(op) % 8
    if op == 0:
        return grid.copy()
    if op == 1:
        return np.rot90(grid, 1)
    if op == 2:
        return np.rot90(grid, 2)
    if op == 3:
        return np.rot90(grid, 3)
    if op == 4:
        return np.fliplr(grid)
    if op == 5:
        return np.flipud(grid)
    if op == 6:
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _validate_generated(grid: np.ndarray) -> bool:
    scene, traces = _trace_scene(grid)
    out = main(grid)
    bg = int(scene["bg"])
    g = scene["grid"]

    assert np.array_equal(out[scene["mirror_mask"]], g[scene["mirror_mask"]]), "mirrors must stay unchanged"
    assert np.array_equal(out[scene["sprite_mask"]], g[scene["sprite_mask"]]), "original sprite must stay unchanged"

    reflected = [trace for trace in traces if trace["final"] != trace["start"]]
    assert reflected, "at least one sprite pixel must reflect"
    lengths = {len(trace["mirror_ids"]) for trace in reflected}
    assert lengths == {1, 2}, "generated examples must show both one-step and relay reflections"
    assert sum(len(trace["mirror_ids"]) == 2 for trace in reflected) >= 2
    assert sum(len(trace["mirror_ids"]) == 1 for trace in reflected) >= 2

    landing_cells = {trace["final"] for trace in reflected}
    assert len(landing_cells) == len(reflected), "generated reflections must stay injective"
    for trace in reflected:
        fr, fc = trace["final"]
        assert int(g[fr, fc]) == bg
        assert int(out[fr, fc]) == int(trace["color"])
        for intermediate in trace["stops"][:-1]:
            ir, ic = intermediate
            assert int(g[ir, ic]) == bg
            assert int(out[ir, ic]) == bg, "intermediate relay images must not be drawn"

    added_mask = (out != bg) & (g == bg)
    assert int(np.count_nonzero(added_mask)) == len(reflected), "output must add exactly the final reflected pixels"

    mirror_lengths = {int(mirror["length"]) for mirror in scene["mirrors"]}
    assert len(mirror_lengths) >= 2, "mirror lengths should visibly vary"
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(BG_CHOICES))
    h = random.randint(5, 6)
    w = random.randint(4, 5)
    chain_rows = random.randint(2, h - 2)
    gap_right = random.randint(2, 3)
    gap_up = random.randint(2, 3)
    gap_left = random.randint(2, 3)

    min_h = 2 * gap_up + chain_rows + h + 2
    min_w = 3 * w + 2 * gap_right + 2 * gap_left
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, min(30, min_h + 6))
    grid_w = random.randint(min_w, min(30, min_w + 5))

    top_min = 2 * gap_up + chain_rows
    top_max = grid_h - h - 2
    left_min = 2 * gap_left + w
    left_max = grid_w - 2 * w - 2 * gap_right
    if top_min > top_max or left_min > left_max:
        return None

    sprite_top = random.randint(top_min, top_max)
    sprite_left = random.randint(left_min, left_max)
    sprite_bottom = sprite_top + h - 1
    sprite_right = sprite_left + w - 1

    mask = _random_sprite_mask(h, w)
    sprite = _random_sprite_array(mask, bg)
    if len({int(v) for v in sprite[mask]}) < 2:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for r, c in np.argwhere(mask):
        grid[sprite_top + r, sprite_left + c] = int(sprite[r, c])

    right_mirror_col = sprite_right + gap_right
    left_mirror_col = sprite_left - gap_left
    upper_mirror_row = sprite_top - gap_up
    right_image_left = right_mirror_col + gap_right

    bottom_rows = list(range(sprite_top + chain_rows, sprite_bottom + 1))
    if len(bottom_rows) < 2:
        return None

    if np.any(grid[sprite_top : sprite_top + chain_rows, right_mirror_col] != bg):
        return None
    if np.any(grid[bottom_rows, left_mirror_col] != bg):
        return None
    if np.any(grid[upper_mirror_row, right_image_left : right_image_left + w] != bg):
        return None

    grid[sprite_top : sprite_top + chain_rows, right_mirror_col] = MIRROR_COLOR
    grid[bottom_rows, left_mirror_col] = MIRROR_COLOR
    grid[upper_mirror_row, right_image_left : right_image_left + w] = MIRROR_COLOR

    transformed = _apply_dihedral(grid, random.randrange(8))
    if _validate_generated(transformed):
        return transformed
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-relay ARC example")
