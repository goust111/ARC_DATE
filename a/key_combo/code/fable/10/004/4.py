from __future__ import annotations

import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
colors_present = _ARC_UTILS.colors_present
components_mask = _ARC_UTILS.components_mask
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location


FRAME_CAVITY_POOL = [
    (3, 5),
    (4, 4),
    (3, 6),
    (5, 4),
    (6, 4),
    (5, 5),
    (4, 7),
    (5, 6),
    (4, 8),
    (5, 7),
    (6, 6),
    (5, 8),
    (7, 6),
    (6, 8),
]
FRAME_COLORS = [c for c in Color.NOT_BLACK if c != Color.GREY]
FRAME_COUNT_RANGE = (4, 5)
OBJECT_COUNT_RANGE = (4, 6)
GRID_SIZE_RANGE = (22, 30)
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80


def _frame_fits_bbox(box_h: int, box_w: int, frame: dict) -> tuple[bool, bool]:
    direct = box_h <= frame["cavity_h"] and box_w <= frame["cavity_w"]
    rotated = box_w <= frame["cavity_h"] and box_h <= frame["cavity_w"]
    return direct, rotated


def _frame_patch(frame: dict) -> np.ndarray:
    h = frame["cavity_h"] + 2
    w = frame["cavity_w"] + 2
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = frame["color"]
    patch[-1, :] = frame["color"]
    patch[:, 0] = frame["color"]
    patch[:, -1] = frame["color"]
    return patch


def _polyline(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    path = [(r, c)]
    while (r, c) != b:
        moves = []
        if r != b[0]:
            moves.append((1 if b[0] > r else -1, 0))
        if c != b[1]:
            moves.append((0, 1 if b[1] > c else -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        path.append((r, c))
    return path


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _shape_with_bbox(box_h: int, box_w: int, target_area: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        hub = (random.randrange(box_h), random.randrange(box_w))
        anchors = {
            (0, random.randrange(box_w)),
            (box_h - 1, random.randrange(box_w)),
            (random.randrange(box_h), 0),
            (random.randrange(box_h), box_w - 1),
        }
        cells: set[tuple[int, int]] = set()
        for anchor in anchors:
            cells.update(_polyline(anchor, hub))

        if len(cells) > target_area:
            continue

        while len(cells) < target_area:
            frontier = {
                nxt
                for r, c in cells
                for nxt in _neighbors4(r, c, box_h, box_w)
                if nxt not in cells
            }
            if not frontier:
                break
            frontier_list = sorted(frontier, key=lambda rc: (rc[0] in (0, box_h - 1)) + (rc[1] in (0, box_w - 1)))
            cells.add(random.choice(frontier_list[: max(1, len(frontier_list) // 2)]))

        if len(cells) != target_area:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if min(rs) != 0 or max(rs) != box_h - 1 or min(cs) != 0 or max(cs) != box_w - 1:
            continue

        patch = np.zeros((box_h, box_w), dtype=int)
        for r, c in cells:
            patch[r, c] = Color.GREY
        if patch.sum() == Color.GREY * box_h * box_w:
            continue
        return patch
    return None


def _parse_frames(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    frames = []
    for color in colors_present(grid, exclude={bg, Color.GREY}):
        comps = components_mask(grid == color, connectivity=4)
        assert len(comps) == 1, "every gauge color must appear as one component"
        coords = comps[0]
        r0, c0, r1, c1 = bbox_coords(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == color
        assert mask.shape[0] >= 3 and mask.shape[1] >= 3, "frames need an interior cavity"
        border = np.zeros_like(mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(mask, border), "non-grey foreground components must be hollow rectangles"
        frames.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cavity_h": int(mask.shape[0] - 2),
                "cavity_w": int(mask.shape[1] - 2),
                "area": int((mask.shape[0] - 2) * (mask.shape[1] - 2)),
            }
        )
    assert frames, "expected at least one gauge frame"
    return frames


def _best_frame_for_bbox(box_h: int, box_w: int, frames: list[dict]) -> tuple[dict | None, bool, bool]:
    fitting = []
    for frame in frames:
        direct, rotated = _frame_fits_bbox(box_h, box_w, frame)
        if direct or rotated:
            fitting.append((frame, direct, rotated))
    if not fitting:
        return None, False, False
    min_area = min(item[0]["area"] for item in fitting)
    best = [item for item in fitting if item[0]["area"] == min_area]
    assert len(best) == 1, "the tightest fitting gauge must be unique"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray object by the smallest hollow-frame gauge whose cavity can contain its bounding box.

    Input structure:
    - A black background contains several one-pixel-thick hollow rectangular frames, each in a distinct non-gray color.
    - Separate gray 4-connected objects sit away from the frames and from one another.
    - The frames have different cavity heights and widths, including both wide and tall cases.

    Rule:
    - For each gray object, take the height and width of its axis-aligned bounding box and allow swapping those two numbers.
    - A frame can contain the object exactly when one orientation of that bounding box fits within the frame cavity in both dimensions.
    - Recolor the whole gray object to the color of the fitting frame with the smallest cavity area; if no frame fits, erase the object to background. Erase all frames in every case.

    Key insight:
    - The hollow rectangles are only size gauges: classification depends on two-dimensional cavity fit, with optional 90-degree rotation, not on contact or on area alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    frames = _parse_frames(g)
    gray_objects = components_mask(g == Color.GREY, connectivity=4)
    assert gray_objects, "expected at least one gray object"

    out = np.full_like(g, bg)
    for coords in gray_objects:
        r0, c0, r1, c1 = bbox_coords(coords)
        box_h = int(r1 - r0 + 1)
        box_w = int(c1 - c0 + 1)
        best, _direct, _rotated = _best_frame_for_bbox(box_h, box_w, frames)
        if best is None:
            continue
        out[coords[:, 0], coords[:, 1]] = best["color"]
    return out


def _sample_frames() -> list[dict] | None:
    count = random.randint(*FRAME_COUNT_RANGE)
    cavities = random.sample(FRAME_CAVITY_POOL, count)
    if not any(h > w for h, w in cavities):
        return None
    if not any(w > h for h, w in cavities):
        return None
    areas = [h * w for h, w in cavities]
    if len(set(areas)) != len(areas):
        return None
    colors = random.sample(FRAME_COLORS, count)
    frames = []
    for idx, ((h, w), color) in enumerate(zip(cavities, colors)):
        frames.append(
            {
                "id": idx,
                "color": int(color),
                "cavity_h": int(h),
                "cavity_w": int(w),
                "area": int(h * w),
            }
        )
    return sorted(frames, key=lambda item: item["area"])


def _candidate_catalog(frames: list[dict]) -> dict[str, list[dict]]:
    max_dim = max(max(frame["cavity_h"], frame["cavity_w"]) for frame in frames) + 1
    rotation = []
    anti = []
    plain = []
    erased = []
    for box_h in range(2, max_dim + 1):
        for box_w in range(2, max_dim + 1):
            best, direct, rotated = _best_frame_for_bbox(box_h, box_w, frames)
            if best is None:
                if any(frame["area"] >= box_h * box_w for frame in frames):
                    erased.append({"bbox": (box_h, box_w)})
                continue

            smaller = []
            for frame in frames:
                if frame["area"] >= best["area"]:
                    continue
                f_direct, f_rotated = _frame_fits_bbox(box_h, box_w, frame)
                if not (f_direct or f_rotated):
                    smaller.append(frame)

            base = {"bbox": (box_h, box_w), "best": best, "direct": direct, "rotated": rotated}
            if rotated and not direct:
                rotation.append(base)
            if direct:
                plain.append(base)
            if any(frame["area"] >= box_h * box_w for frame in smaller):
                anti.append(base)
    return {"rotation": rotation, "anti": anti, "plain": plain, "erased": erased}


def _choose_roles(frames: list[dict]) -> list[dict] | None:
    catalog = _candidate_catalog(frames)
    if not catalog["rotation"] or not catalog["anti"] or not catalog["plain"] or not catalog["erased"]:
        return None

    rotation_choices = random.sample(catalog["rotation"], len(catalog["rotation"]))
    anti_choices = random.sample(catalog["anti"], len(catalog["anti"]))
    plain_choices = random.sample(catalog["plain"], len(catalog["plain"]))

    for rotation in rotation_choices:
        for anti in anti_choices:
            if anti["bbox"] == rotation["bbox"]:
                continue
            for plain in plain_choices:
                if plain["bbox"] in {rotation["bbox"], anti["bbox"]}:
                    continue
                survivor_frames = {rotation["best"]["color"], anti["best"]["color"], plain["best"]["color"]}
                if len(survivor_frames) < 2:
                    continue
                erased = random.choice(catalog["erased"])
                roles = [
                    {"role": "rotation", **rotation},
                    {"role": "anti", **anti},
                    {"role": "plain", **plain},
                    {"role": "erased", **erased, "best": None, "direct": False, "rotated": False},
                ]
                extra_budget = random.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1]) - len(roles)
                extra_plain = [item for item in plain_choices if item["bbox"] not in {r["bbox"] for r in roles}]
                extra_rotation = [item for item in rotation_choices if item["bbox"] not in {r["bbox"] for r in roles}]
                for _ in range(extra_budget):
                    pool = extra_plain if extra_plain and random.random() < 0.7 else extra_rotation
                    if not pool:
                        pool = extra_plain or extra_rotation
                    if not pool:
                        break
                    picked = pool.pop()
                    roles.append({"role": "extra", **picked})
                return roles
    return None


def _shape_area_target(box_h: int, box_w: int, role: str) -> int:
    box_area = box_h * box_w
    if role in {"anti", "erased"}:
        lo = max(4, (box_area + 2) // 3)
        hi = max(lo, box_area - max(2, box_area // 4))
    elif role == "rotation":
        lo = max(4, box_area // 2)
        hi = box_area - 1
    else:
        lo = max(4, box_area // 2)
        hi = box_area
    return random.randint(lo, hi)


def _build_objects(role_specs: list[dict], frames: list[dict]) -> list[dict] | None:
    objects = []
    for spec in role_specs:
        box_h, box_w = spec["bbox"]
        patch = None
        for _ in range(SHAPE_ATTEMPTS):
            target_area = _shape_area_target(box_h, box_w, spec["role"])
            patch = _shape_with_bbox(box_h, box_w, target_area)
            if patch is None:
                continue
            actual_area = int((patch == Color.GREY).sum())
            if spec["role"] in {"anti", "erased"}:
                nonfitting = [
                    frame
                    for frame in frames
                    if frame["area"] >= actual_area
                    and not any(_frame_fits_bbox(box_h, box_w, frame))
                ]
                if not nonfitting:
                    patch = None
                    continue
            break
        if patch is None:
            return None
        objects.append({**spec, "patch": patch, "area": int((patch == Color.GREY).sum())})
    return objects


def _place_scene(frames: list[dict], objects: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]] | None:
    frame_patches = [_frame_patch(frame) for frame in frames]
    object_patches = [obj["patch"] for obj in objects]
    total_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in frame_patches + object_patches)
    min_side = max(GRID_SIZE_RANGE[0], int(np.ceil(np.sqrt(total_area * 1.8))))
    if min_side > GRID_SIZE_RANGE[1]:
        return None

    height = random.randint(min_side, GRID_SIZE_RANGE[1])
    width = random.randint(min_side, GRID_SIZE_RANGE[1])
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(grid)
    footprint = np.zeros_like(grid)

    placed_frames = []
    for frame, patch in zip(frames, frame_patches):
        block = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                footprint,
                block,
                transparent=0,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(footprint, block, r0, c0, transparent=0)
        paste(grid, patch, r0, c0)
        placed_frames.append({**frame, "bbox": (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)})

    placed_objects = []
    for obj, patch in zip(objects, object_patches):
        block = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                footprint,
                block,
                transparent=0,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(footprint, block, r0, c0, transparent=0)
        paste(grid, patch, r0, c0, transparent=0)
        expected_patch = np.zeros_like(patch)
        if obj["best"] is not None:
            expected_patch[patch == Color.GREY] = obj["best"]["color"]
        paste(expected, expected_patch, r0, c0, transparent=0)
        placed_objects.append({**obj, "bbox": (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)})

    return grid, expected, placed_frames, placed_objects


def _validate_scene(
    grid: np.ndarray,
    expected: np.ndarray,
    placed_frames: list[dict],
    placed_objects: list[dict],
) -> bool:
    parsed_frames = sorted(_parse_frames(grid), key=lambda item: item["color"])
    expected_frames = sorted(
        [
            {
                "color": int(frame["color"]),
                "cavity_h": int(frame["cavity_h"]),
                "cavity_w": int(frame["cavity_w"]),
                "area": int(frame["area"]),
            }
            for frame in placed_frames
        ],
        key=lambda item: item["color"],
    )
    if [
        (item["color"], item["cavity_h"], item["cavity_w"], item["area"])
        for item in parsed_frames
    ] != [
        (item["color"], item["cavity_h"], item["cavity_w"], item["area"])
        for item in expected_frames
    ]:
        return False

    seen_roles = set()
    survivor_colors = set()
    for obj in placed_objects:
        r0, c0, r1, c1 = obj["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.array_equal(patch, obj["patch"]):
            return False
        box_h, box_w = obj["bbox"][2] - obj["bbox"][0] + 1, obj["bbox"][3] - obj["bbox"][1] + 1
        best, direct, rotated = _best_frame_for_bbox(box_h, box_w, parsed_frames)
        actual_area = int((obj["patch"] == Color.GREY).sum())

        if obj["role"] == "erased":
            if best is not None:
                return False
            if not any(frame["area"] >= actual_area and not any(_frame_fits_bbox(box_h, box_w, frame)) for frame in parsed_frames):
                return False
            seen_roles.add("erased")
            continue

        if best is None or best["color"] != obj["best"]["color"]:
            return False
        survivor_colors.add(best["color"])

        if obj["role"] == "rotation":
            if direct or not rotated:
                return False
            seen_roles.add("rotation")
        elif obj["role"] == "anti":
            smaller = [
                frame
                for frame in parsed_frames
                if frame["area"] < best["area"]
                and frame["area"] >= box_h * box_w
                and not any(_frame_fits_bbox(box_h, box_w, frame))
            ]
            if not smaller:
                return False
            if not any(frame["area"] >= actual_area for frame in smaller):
                return False
            seen_roles.add("anti")
        elif obj["role"] == "plain":
            if not direct:
                return False
            seen_roles.add("plain")

    if seen_roles != {"rotation", "anti", "plain", "erased"}:
        return False
    if len(survivor_colors) < 2:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    frames = _sample_frames()
    if frames is None:
        return None
    role_specs = _choose_roles(frames)
    if role_specs is None:
        return None
    objects = _build_objects(role_specs, frames)
    if objects is None:
        return None
    scene = _place_scene(frames, objects)
    if scene is None:
        return None
    grid, expected, placed_frames, placed_objects = scene
    if not _validate_scene(grid, expected, placed_frames, placed_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hollow-gauge ARC input")
