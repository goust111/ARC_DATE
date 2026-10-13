from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, components_mask, crop_bbox, find_components, get_bg, in_bounds, neighbors4


BACKGROUND = int(Color.BLACK)
BASELINE = int(Color.BLUE)
SOURCE_COLOR = int(Color.GREEN)
MIRROR_COLOR = int(Color.RED)
COLLISION_COLOR = int(Color.YELLOW)

GRID_ROWS = (14, 18)
GRID_COLS = (17, 22)
OBJECT_COUNTS = (3, 4)
FRAME_OPTIONS = ((3, 3), (3, 4), (4, 3), (4, 4))
ALL_SIDES = ("left", "right", "top", "bottom")
INPUT_GAP = 2
MAX_ATTEMPTS = 500


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_outline(patch: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    patch[top, left : right + 1] = SOURCE_COLOR
    patch[bottom, left : right + 1] = SOURCE_COLOR
    patch[top : bottom + 1, left] = SOURCE_COLOR
    patch[top : bottom + 1, right] = SOURCE_COLOR


def _make_patch(side: str, frame_h: int, frame_w: int) -> np.ndarray:
    if side in {"left", "right"}:
        patch = np.full((frame_h, frame_w + 1), BACKGROUND, dtype=int)
        frame_left = 1 if side == "left" else 0
        tab_col = 0 if side == "left" else frame_w
        _draw_outline(patch, 0, frame_left, frame_h, frame_w)
        patch[frame_h // 2, tab_col] = SOURCE_COLOR
        return patch

    patch = np.full((frame_h + 1, frame_w), BACKGROUND, dtype=int)
    frame_top = 1 if side == "top" else 0
    tab_row = 0 if side == "top" else frame_h
    _draw_outline(patch, frame_top, 0, frame_h, frame_w)
    patch[tab_row, frame_w // 2] = SOURCE_COLOR
    return patch


def _tab_side(mask: np.ndarray) -> str:
    h, w = mask.shape
    counts = {
        "top": int(mask[0, :].sum()),
        "bottom": int(mask[h - 1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, w - 1].sum()),
    }

    singleton = [side for side, count in counts.items() if count == 1]
    if len(singleton) == 1:
        return singleton[0]

    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    candidates = []
    for side in ALL_SIDES:
        if counts[side] == counts[opposite[side]] + 1:
            candidates.append(side)
    if len(candidates) == 1:
        return candidates[0]

    raise AssertionError("expected a unique tab side")


def _reflected_mask_and_origin(mask: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = mask.shape
    if side in {"left", "right"}:
        reflected = np.fliplr(mask)
        top = r0
        left = c1 + 2 if side == "left" else c0 - w - 1
        return reflected, top, left

    reflected = np.flipud(mask)
    top = r1 + 2 if side == "top" else r0 - h - 1
    return reflected, top, c0


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _analyze(grid: np.ndarray, *, with_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    bg = int(get_bg(g))
    baseline_row = g[height - 1, :]
    baseline_colors = set(map(int, np.unique(baseline_row)))
    assert len(baseline_colors) == 1, "expected a uniform baseline row"
    baseline = int(next(iter(baseline_colors)))

    comps = find_components(g, bg=bg, colors=[SOURCE_COLOR], connectivity=4, monochromatic=True)
    assert comps, "expected green outline objects"

    column_entries: list[list[tuple[int, int]]] = [[] for _ in range(width)]
    source_reflections: dict[int, list[tuple[int, int]]] = {}
    occupied_reflections: set[tuple[int, int]] = set()
    sides: list[str] = []

    for source_id, comp in enumerate(comps):
        patch = crop_bbox(g, comp["bbox"])
        mask = patch == SOURCE_COLOR
        side = _tab_side(mask)
        sides.append(side)
        reflected, top, left = _reflected_mask_and_origin(mask, tuple(map(int, comp["bbox"])), side)

        coords: list[tuple[int, int]] = []
        for dr, dc in np.argwhere(reflected):
            rr = int(top + dr)
            cc = int(left + dc)
            assert in_bounds(g.shape, rr, cc), "mirrored copy left the grid"
            assert int(g[rr, cc]) == bg, "mirrored copy must land on background"
            assert (rr, cc) not in occupied_reflections, "mirrored copies must not overlap"
            occupied_reflections.add((rr, cc))
            coords.append((rr, cc))
            column_entries[cc].append((rr, source_id))
        source_reflections[source_id] = coords

    settled_by_source = {source_id: [] for source_id in source_reflections}
    for col in range(width):
        ordered = sorted(column_entries[col], key=lambda item: item[0])
        count = len(ordered)
        for idx, (_row, source_id) in enumerate(ordered):
            dest_row = height - count + idx
            settled_by_source[source_id].append((dest_row, col))

    source_components = []
    for source_id, coords in settled_by_source.items():
        mask = np.zeros((height, width), dtype=bool)
        for rr, cc in coords:
            mask[rr, cc] = True
        for comp_coords in components_mask(mask, connectivity=4):
            coord_list = [tuple(map(int, cell)) for cell in comp_coords]
            source_components.append(
                {
                    "source": int(source_id),
                    "coords": coord_list,
                    "coord_set": set(coord_list),
                    "area": len(coord_list),
                }
            )

    smaller_components: set[int] = set()
    collision_pairs = []
    for i in range(len(source_components)):
        for j in range(i + 1, len(source_components)):
            comp_a = source_components[i]
            comp_b = source_components[j]
            if comp_a["source"] == comp_b["source"]:
                continue

            touching = False
            for rr, cc in comp_a["coords"]:
                for nr, nc in neighbors4(rr, cc):
                    if (int(nr), int(nc)) in comp_b["coord_set"]:
                        touching = True
                        break
                if touching:
                    break
            if not touching:
                continue

            collision_pairs.append((i, j))
            if comp_a["area"] < comp_b["area"]:
                smaller_components.add(i)
            elif comp_b["area"] < comp_a["area"]:
                smaller_components.add(j)

    out = np.full_like(g, bg)
    out[height - 1, :] = baseline
    for coords in settled_by_source.values():
        for rr, cc in coords:
            out[rr, cc] = MIRROR_COLOR
    for comp_index in smaller_components:
        for rr, cc in source_components[comp_index]["coords"]:
            out[rr, cc] = COLLISION_COLOR

    if with_details:
        return out, {
            "sides": sides,
            "source_components": source_components,
            "collision_pairs": collision_pairs,
            "settled_by_source": settled_by_source,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each green tabbed outline to the side opposite its tab, erase the originals, let the copies fall, and recolor the smaller side of each collision.

    Input structure:
    - The grid has a black background with a uniform blue baseline row at the bottom.
    - Above the baseline are 3-4 separate 4-connected green hollow outline objects.
    - Each green outline has exactly one one-pixel tab on one side of its bounding box.

    Rule:
    - Detect the tab side of each green outline.
    - Create a red mirrored copy on the side opposite the tab with a one-cell gap: left/right tab means a horizontal mirror, top/bottom tab means a vertical mirror.
    - Erase the original green outlines, then compact all red pixels downward within their columns, allowing them to settle onto the baseline row.
    - After settling, compare 4-connected source components from different originals; whenever two such components touch orthogonally, recolor the smaller one yellow and keep the larger one red.

    Key insight:
    - The tab determines where the mirrored copy starts, that starting column pattern determines how gravity stacks the copy, and those settled column stacks determine which source component becomes the smaller collision loser.
    """
    return _analyze(grid, with_details=False)


def _green_from_red(red_top: int, red_left: int, patch: np.ndarray, side: str) -> tuple[int, int]:
    h, w = patch.shape
    if side == "left":
        return red_top, red_left - w - 1
    if side == "right":
        return red_top, red_left + w + 1
    if side == "top":
        return red_top - h - 1, red_left
    return red_top + h + 1, red_left


def _patch_info(side: str) -> dict:
    frame_h, frame_w = random.choice(FRAME_OPTIONS)
    patch = _make_patch(side, frame_h, frame_w)
    mask = patch == SOURCE_COLOR
    return {
        "side": side,
        "patch": patch,
        "mask": mask,
        "shape": patch.shape,
        "area": int(mask.sum()),
    }


def _random_pair_red_positions(width: int, info_a: dict, info_b: dict) -> tuple[int, int] | None:
    wa = int(info_a["shape"][1])
    wb = int(info_b["shape"][1])
    total = wa + wb
    if total >= width - 2:
        return None
    left_a = random.randint(1, width - total - 1)
    if random.random() < 0.5:
        left_b = left_a + wa
    else:
        left_b = left_a
        left_a = left_b + wb
    return left_a, left_b


def _green_bbox_for(obj: dict) -> tuple[int, int, int, int] | None:
    r0 = int(obj["green_top"])
    c0 = int(obj["green_left"])
    h, w = obj["shape"]
    if r0 < 0 or c0 < 0:
        return None
    return r0, c0, r0 + h - 1, c0 + w - 1


def _fits_green(obj: dict, shape: tuple[int, int], placed_bboxes: list[tuple[int, int, int, int]]) -> bool:
    height, width = shape
    bbox = _green_bbox_for(obj)
    if bbox is None:
        return False
    r0, c0, r1, c1 = bbox
    if r1 >= height - 1 or c1 >= width:
        return False
    for other_bbox in placed_bboxes:
        if not _bbox_separated(bbox, other_bbox, INPUT_GAP):
            return False
    return True


def _assign_layout(objects: list[dict], pair_a: int, pair_b: int, shape: tuple[int, int]) -> bool:
    height, width = shape
    top_limit = max(2, height // 2 - 2)

    for _ in range(120):
        pair_lefts = _random_pair_red_positions(width, objects[pair_a], objects[pair_b])
        if pair_lefts is None:
            return False
        objects[pair_a]["red_top"] = random.randint(1, top_limit)
        objects[pair_b]["red_top"] = random.randint(1, top_limit)
        objects[pair_a]["red_left"], objects[pair_b]["red_left"] = pair_lefts
        objects[pair_a]["green_top"], objects[pair_a]["green_left"] = _green_from_red(
            int(objects[pair_a]["red_top"]),
            int(objects[pair_a]["red_left"]),
            objects[pair_a]["patch"],
            objects[pair_a]["side"],
        )
        objects[pair_b]["green_top"], objects[pair_b]["green_left"] = _green_from_red(
            int(objects[pair_b]["red_top"]),
            int(objects[pair_b]["red_left"]),
            objects[pair_b]["patch"],
            objects[pair_b]["side"],
        )

        placed_bboxes = []
        if not _fits_green(objects[pair_a], shape, placed_bboxes):
            continue
        placed_bboxes.append(_green_bbox_for(objects[pair_a]))
        if not _fits_green(objects[pair_b], shape, placed_bboxes):
            continue
        placed_bboxes.append(_green_bbox_for(objects[pair_b]))

        remaining = [idx for idx in range(len(objects)) if idx not in {pair_a, pair_b}]
        ok = True
        for idx in remaining:
            obj = objects[idx]
            h, w = obj["shape"]
            max_left = width - w - 1
            if max_left < 1:
                ok = False
                break

            placed = False
            for _ in range(100):
                obj["red_top"] = random.randint(1, top_limit)
                obj["red_left"] = random.randint(1, max_left)
                obj["green_top"], obj["green_left"] = _green_from_red(
                    int(obj["red_top"]),
                    int(obj["red_left"]),
                    obj["patch"],
                    obj["side"],
                )
                if not _fits_green(obj, shape, placed_bboxes):
                    continue
                placed_bboxes.append(_green_bbox_for(obj))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return True

    return False


def _build_input(objects: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    height, width = shape
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[height - 1, :] = BASELINE

    bboxes = []
    for obj in objects:
        patch = obj["patch"]
        r0 = obj["green_top"]
        c0 = obj["green_left"]
        h, w = patch.shape
        if r0 < 0 or c0 < 0 or r0 + h > height - 1 or c0 + w > width:
            return None
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if bbox[2] >= height - 1:
            return None
        for other_bbox in bboxes:
            if not _bbox_separated(bbox, other_bbox, INPUT_GAP):
                return None
        patch_mask = patch == SOURCE_COLOR
        target = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any(target[patch_mask] != BACKGROUND):
            return None
        target[patch_mask] = SOURCE_COLOR
        bboxes.append(bbox)
    return grid


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_ROWS)
    width = random.randint(*GRID_COLS)
    count = random.choice(OBJECT_COUNTS)

    sides = [random.choice(("left", "right")), random.choice(("top", "bottom"))]
    while len(sides) < count:
        sides.append(random.choice(ALL_SIDES))
    random.shuffle(sides)

    objects = [_patch_info(side) for side in sides]
    pair_a, pair_b = random.sample(range(count), 2)
    for _ in range(20):
        if objects[pair_a]["area"] != objects[pair_b]["area"]:
            break
        objects[pair_b] = _patch_info(objects[pair_b]["side"])
    if objects[pair_a]["area"] == objects[pair_b]["area"]:
        return None
    if not _assign_layout(objects, pair_a, pair_b, (height, width)):
        return None

    grid = _build_input(objects, (height, width))
    if grid is None:
        return None

    try:
        out, details = _analyze(grid, with_details=True)
    except AssertionError:
        return None

    if not any(side in {"left", "right"} for side in details["sides"]):
        return None
    if not any(side in {"top", "bottom"} for side in details["sides"]):
        return None
    if not details["collision_pairs"]:
        return None
    for a_idx, b_idx in details["collision_pairs"]:
        area_a = details["source_components"][a_idx]["area"]
        area_b = details["source_components"][b_idx]["area"]
        if area_a == area_b:
            return None
    if not np.any(out == MIRROR_COLOR):
        return None
    if not np.any(out == COLLISION_COLOR):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
