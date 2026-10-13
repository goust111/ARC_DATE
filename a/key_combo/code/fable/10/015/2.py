from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, find_components


GREY = Color.GREY
BACKGROUND = Color.BLACK
COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
FAMILIES = ("line", "elbow", "stair", "dogleg")

GRID_SIDE_RANGE = (20, 30)
PAIR_COUNT_RANGE = (2, 4)
DEPTH_OPTIONS = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _peel_depth(mask: np.ndarray) -> int:
    """Number of 4-neighbor boundary peels needed to erase a solid blob."""
    cur = np.asarray(mask, dtype=bool).copy()
    depth = 0
    while cur.any():
        depth += 1
        nxt = np.zeros_like(cur)
        if cur.shape[0] >= 3 and cur.shape[1] >= 3:
            nxt[1:-1, 1:-1] = (
                cur[1:-1, 1:-1]
                & cur[:-2, 1:-1]
                & cur[2:, 1:-1]
                & cur[1:-1, :-2]
                & cur[1:-1, 2:]
            )
        cur = nxt
    return depth


def _mask_has_hole(mask: np.ndarray) -> bool:
    bg = ~np.asarray(mask, dtype=bool)
    padded = np.pad(bg, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(padded & ~seen))


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _polyline_path(steps: list[int], start_horizontal: bool) -> list[tuple[int, int]]:
    r = c = 0
    coords = [(r, c)]
    horizontal = start_horizontal
    for step_count in steps:
        for _ in range(step_count):
            if horizontal:
                c += 1
            else:
                r += 1
            coords.append((r, c))
        horizontal = not horizontal
    return coords


def _path_for_family(family: str, length_mode: str) -> list[tuple[int, int]]:
    if length_mode == "short":
        step_lo, step_hi = 1, 2
        line_len = (1, 3)
    elif length_mode == "long":
        step_lo, step_hi = 2, 5
        line_len = (8, 14)
    else:
        step_lo, step_hi = 1, 4
        line_len = (4, 9)

    if family == "line":
        length = random.randint(*line_len)
        return _polyline_path([length - 1], start_horizontal=random.choice([True, False]))

    if family == "elbow":
        return _polyline_path(
            [random.randint(max(1, step_lo), step_hi + 1), random.randint(max(1, step_lo), step_hi + 1)],
            start_horizontal=random.choice([True, False]),
        )

    if family == "stair":
        seg_count = random.randint(3, 5 if length_mode != "short" else 4)
        steps = [random.randint(step_lo, step_hi) for _ in range(seg_count)]
        return _polyline_path(steps, start_horizontal=random.choice([True, False]))

    if family == "dogleg":
        base = [random.randint(max(1, step_lo), step_hi + 1), random.randint(1, step_hi), random.randint(max(1, step_lo), step_hi + 1)]
        return _polyline_path(base, start_horizontal=random.choice([True, False]))

    raise ValueError(f"unknown family: {family}")


def _dilate_path(path: list[tuple[int, int]], radius: int) -> np.ndarray:
    pts: set[tuple[int, int]] = set()
    for pr, pc in path:
        for dr in range(-radius, radius + 1):
            rem = radius - abs(dr)
            for dc in range(-rem, rem + 1):
                pts.add((pr + dr, pc + dc))
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return mask


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = np.asarray(mask, dtype=np.uint8)
    return cropped.shape, cropped.tobytes()


def _make_shape(depth: int, *, family: str | None = None, length_mode: str = "medium", avoid_signature: tuple[tuple[int, int], bytes] | None = None) -> np.ndarray | None:
    for _ in range(120):
        chosen_family = family or random.choice(FAMILIES)
        mask = _dilate_path(_path_for_family(chosen_family, length_mode), depth - 1)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randint(0, 3))
        if _peel_depth(mask) != depth:
            continue
        if _mask_has_hole(mask):
            continue
        sig = _shape_signature(mask)
        if avoid_signature is not None and sig == avoid_signature:
            continue
        return mask
    return None


def _expand8(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _place_patches(shape: tuple[int, int], patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    occupied = np.zeros(shape, dtype=bool)
    blocked = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: int(patches[idx].sum()), reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates = []
        for r0 in range(shape[0] - ph + 1):
            for c0 in range(shape[1] - pw + 1):
                if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & patch):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch
        blocked = _expand8(occupied)

    return [(int(r), int(c)) for r, c in placements]


def _render_case(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for obj in objects:
        r0, c0 = obj["pos"]
        mask = obj["mask"]
        h, w = mask.shape
        view = grid[r0 : r0 + h, c0 : c0 + w]
        view[mask] = int(obj["color"])
    return grid


def _analyze_case(grid: np.ndarray) -> tuple[list[dict], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9
    assert int(np.bincount(g.ravel(), minlength=10).argmax()) == BACKGROUND

    comps = find_components(g, bg=BACKGROUND, connectivity=4, monochromatic=True, return_format="dict")
    colored: list[dict] = []
    grays: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        mask = _component_mask(comp)
        info = {
            "color": color,
            "coords": comp["coords"],
            "bbox": tuple(map(int, comp["bbox"])),
            "area": int(comp["area"]),
            "depth": _peel_depth(mask),
            "shape": mask.shape,
        }
        if color == GREY:
            grays.append(info)
        else:
            colored.append(info)

    assert len(colored) == len(grays) >= 1

    color_by_depth: dict[int, int] = {}
    for obj in colored:
        assert obj["depth"] not in color_by_depth
        color_by_depth[obj["depth"]] = obj["color"]

    gray_depths = [obj["depth"] for obj in grays]
    assert len(set(gray_depths)) == len(grays)
    assert set(gray_depths) == set(color_by_depth)

    expected = g.copy()
    for obj in grays:
        expected[obj["coords"][:, 0], obj["coords"][:, 1]] = color_by_depth[obj["depth"]]

    return colored, grays, expected


def _has_area_depth_counterexample(colored: list[dict], grays: list[dict]) -> bool:
    all_objs = colored + grays
    for a in all_objs:
        for b in all_objs:
            if a["depth"] < b["depth"] and a["area"] > b["area"]:
                return True
    return False


def _paired_visible_match(colored: list[dict], grays: list[dict]) -> bool:
    by_depth = {obj["depth"]: obj for obj in colored}
    for obj in grays:
        peer = by_depth[obj["depth"]]
        if obj["area"] == peer["area"] or obj["shape"] == peer["shape"]:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every gray blob with the color of the equally deep colored blob under repeated 4-neighbor peeling.

    Input structure:
    - A black background contains several separated solid blobs.
    - Some blobs are gray; the rest are already colored, with one distinct color per colored blob.
    - Within the colored blobs, peel depths are all different; within the gray blobs, peel depths are all different; the two depth sets are equal.

    Rule:
    - For each blob, repeatedly remove the current outer layer of cells that 4-touch the background until the blob disappears.
    - The number of peel rounds is that blob's depth.
    - Recolor each gray blob uniformly to the color of the colored blob with the same depth, while leaving the colored blobs unchanged.

    Key insight:
    - Pairing depends on hidden thickness under erosion depth, not on area, bounding box, or outline similarity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert int(np.bincount(g.ravel(), minlength=10).argmax()) == BACKGROUND, "expected black background"

    _colored, _grays, expected = _analyze_case(g)
    return expected


def _build_depth_pair(depth: int, pair_color: int, emphatic: str | None) -> tuple[dict, dict] | None:
    primary_family, secondary_family = random.sample(FAMILIES, 2)
    color_mode = "medium"
    gray_mode = "medium"
    if emphatic == "long":
        color_mode, gray_mode = "long", random.choice(["medium", "long"])
    elif emphatic == "short":
        color_mode, gray_mode = "short", random.choice(["short", "medium"])
    else:
        color_mode, gray_mode = random.sample(["short", "medium", "long"], 2)

    color_mask = _make_shape(depth, family=primary_family, length_mode=color_mode)
    if color_mask is None:
        return None

    gray_mask = None
    avoid_sig = _shape_signature(color_mask)
    for _ in range(80):
        candidate = _make_shape(depth, family=secondary_family, length_mode=gray_mode, avoid_signature=avoid_sig)
        if candidate is None:
            continue
        if candidate.sum() != color_mask.sum() and candidate.shape != color_mask.shape:
            gray_mask = candidate
            break
    if gray_mask is None:
        return None

    return (
        {"depth": depth, "color": int(pair_color), "mask": color_mask, "role": "colored"},
        {"depth": depth, "color": GREY, "mask": gray_mask, "role": "gray"},
    )


def _gen_attempt() -> np.ndarray | None:
    side_h = random.randint(*GRID_SIDE_RANGE)
    side_w = random.randint(*GRID_SIDE_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    depths = sorted(random.sample(DEPTH_OPTIONS, pair_count))
    pair_colors = random.sample(COLOR_POOL, pair_count)

    emphatic_by_depth: dict[int, str | None] = {depth: None for depth in depths}
    if len(depths) >= 2:
        shallow_idx = random.randint(0, len(depths) - 2)
        deep_idx = random.randint(shallow_idx + 1, len(depths) - 1)
        emphatic_by_depth[depths[shallow_idx]] = "long"
        emphatic_by_depth[depths[deep_idx]] = "short"

    objects: list[dict] = []
    for depth, pair_color in zip(depths, pair_colors):
        pair = _build_depth_pair(depth, pair_color, emphatic_by_depth[depth])
        if pair is None:
            return None
        objects.extend(pair)

    random.shuffle(objects)
    patches = [obj["mask"] for obj in objects]
    placements = _place_patches((side_h, side_w), patches)
    if placements is None:
        return None

    for obj, pos in zip(objects, placements):
        obj["pos"] = pos

    grid = _render_case((side_h, side_w), objects)
    try:
        colored, grays, expected = _analyze_case(grid)
    except AssertionError:
        return None

    if not (2 <= len(colored) <= 4):
        return None
    if len({obj["color"] for obj in colored}) != len(colored):
        return None
    if not _has_area_depth_counterexample(colored, grays):
        return None
    if _paired_visible_match(colored, grays):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid thickness-pairing input")
