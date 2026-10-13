from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared", _UTILS_PATH)
if _UTILS_SPEC is None or _UTILS_SPEC.loader is None:
    raise ImportError(f"cannot load arc utils from {_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BASE_SHAPE = (14, 14)
FOUR_CORNERS = [(0, 0), (9, 9), (4, 13), (13, 4)]
SIX_EXTRA_CORNERS = [(2, 7), (10, 3)]
FILLER_A = [(3, 2), (4, 2), (5, 2)]
FILLER_B = [(11, 6), (11, 7), (11, 8)]
SIX_FILLERS = [
    [(2, 4), (3, 4)],
    [(2, 5), (3, 5)],
    [(7, 3), (8, 3)],
    [(3, 4), (3, 5)],
    [(2, 4), (3, 4), (3, 5)],
    [(2, 5), (3, 4), (3, 5)],
    [(2, 4), (2, 5), (3, 4)],
    [(2, 4), (2, 5), (3, 5)],
]
TARGET_MASKS = [
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
]
COLOR_POOL = list(range(10))
GEN_ATTEMPTS = 400


def _all_matchings(points: list[tuple[int, int]]) -> list[tuple[tuple[tuple[int, int], tuple[int, int]], ...]]:
    pts = tuple(sorted((int(r), int(c)) for r, c in points))
    out: set[tuple[tuple[tuple[int, int], tuple[int, int]], ...]] = set()

    def rec(rest: list[tuple[int, int]], cur: list[tuple[tuple[int, int], tuple[int, int]]]) -> None:
        if not rest:
            out.add(tuple(sorted(tuple(sorted(pair)) for pair in cur)))
            return
        head = rest[0]
        for i in range(1, len(rest)):
            rec(rest[1:i] + rest[i + 1 :], cur + [(head, rest[i])])

    rec(list(pts), [])
    return sorted(out)


def _rect_from_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if a[0] == b[0] or a[1] == b[1]:
        return None
    return min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])


def _coords_inside_box(coords: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    rr = coords[:, 0]
    cc = coords[:, 1]
    return bool(np.all((r0 <= rr) & (rr <= r1) & (c0 <= cc) & (cc <= c1)))


def _coords_overlap_box(coords: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    rr = coords[:, 0]
    cc = coords[:, 1]
    return bool(np.any((r0 <= rr) & (rr <= r1) & (c0 <= cc) & (cc <= c1)))


def _corner_color_and_points(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    mono = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    singletons_by_color: dict[int, list[tuple[int, int]]] = {}
    area_by_color: dict[int, int] = {}
    for comp in mono:
        color = int(comp["color"])
        area_by_color[color] = area_by_color.get(color, 0) + int(comp["area"])
        if int(comp["area"]) == 1:
            singletons_by_color.setdefault(color, []).append(tuple(map(int, comp["coords"][0])))
    candidates = [
        (color, sorted(points))
        for color, points in singletons_by_color.items()
        if len(points) in (4, 6) and area_by_color.get(color, 0) == len(points)
    ]
    assert len(candidates) == 1
    return candidates[0]


def _sprite_records(grid: np.ndarray, bg: int, corner_color: int) -> list[dict]:
    sprite_mask = (grid != bg) & (grid != corner_color)
    sprites = []
    for coords in components_mask(sprite_mask, connectivity=4):
        assert int(coords.shape[0]) > 1
        sprites.append({"coords": coords, "bbox": bbox_coords(coords)})
    assert sprites
    return sprites


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    corner_color, corner_points = _corner_color_and_points(g, bg)
    sprites = _sprite_records(g, bg, corner_color)

    valid_matchings = []
    all_matchings = _all_matchings(corner_points)
    for matching in all_matchings:
        rects = []
        contained_by_rect = []
        ok = True
        for a, b in matching:
            box = _rect_from_pair(a, b)
            if box is None:
                ok = False
                break
            full = []
            partial = False
            for idx, sprite in enumerate(sprites):
                coords = sprite["coords"]
                inside = _coords_inside_box(coords, box)
                overlap = _coords_overlap_box(coords, box)
                if inside:
                    full.append(idx)
                elif overlap:
                    partial = True
            if partial or not full:
                ok = False
                break
            rects.append(box)
            contained_by_rect.append(tuple(full))
        if ok:
            valid_matchings.append(
                {
                    "matching": matching,
                    "rects": tuple(rects),
                    "contained_by_rect": tuple(contained_by_rect),
                }
            )

    assert len(valid_matchings) == 1
    chosen = valid_matchings[0]
    rects = chosen["rects"]
    inter = (
        max(box[0] for box in rects),
        max(box[1] for box in rects),
        min(box[2] for box in rects),
        min(box[3] for box in rects),
    )
    assert inter[0] <= inter[2] and inter[1] <= inter[3]

    fully_in_intersection = [idx for idx, sprite in enumerate(sprites) if _coords_inside_box(sprite["coords"], inter)]
    assert len(fully_in_intersection) == 1
    target_idx = fully_in_intersection[0]
    target_bbox = sprites[target_idx]["bbox"]

    corner_bbox = bbox_coords(np.array(corner_points, dtype=int))
    fully_in_corner_bbox = [idx for idx, sprite in enumerate(sprites) if _coords_inside_box(sprite["coords"], corner_bbox)]
    assert len(fully_in_corner_bbox) > 1

    return {
        "bg": bg,
        "corner_color": corner_color,
        "corner_points": corner_points,
        "sprites": sprites,
        "rects": rects,
        "contained_by_rect": chosen["contained_by_rect"],
        "intersection": inter,
        "target_idx": target_idx,
        "target_bbox": target_bbox,
        "corner_bbox": corner_bbox,
        "all_matchings": all_matchings,
        "valid_matchings": valid_matchings,
    }


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    if len(info["all_matchings"]) <= 1:
        return False
    if not all(len(contained) > 1 for contained in info["contained_by_rect"]):
        return False
    if len(info["corner_points"]) not in (4, 6):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Disambiguate invisible corner-generated rectangles, intersect them, and crop the unique enclosed sprite.

    Input structure:
    - The dominant color is the background.
    - Several disconnected multi-pixel sprites sit on that background.
    - One separate marker color appears as four or six isolated single-pixel corners, and pairing two such pixels defines an invisible axis-aligned rectangle.

    Rule:
    - Enumerate all perfect pairings of the corner pixels.
    - Keep the unique pairing whose rectangles each fully contain at least one whole sprite and do not cut through any sprite.
    - Intersect those rectangle regions, find the unique sprite fully contained in the intersection, and return its tight bounding-box crop.

    Key insight:
    - The corner pixels are only rectangle generators, and even their correct pairing must be inferred from which invisible rectangles cleanly enclose whole sprites; only the full rectangle intersection isolates the output sprite.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    return crop_bbox(g, info["target_bbox"])


def _transform_mask(mask: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, rot_k)
    if flip_lr:
        out = np.fliplr(out)
    return out


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for rot_k in range(4):
        for flip_lr in (False, True):
            cur = _transform_mask(mask, rot_k, flip_lr)
            key = tuple(cur.ravel().tolist()) + cur.shape
            if key not in seen:
                seen.add(key)
                variants.append(cur)
    return variants


def _palette(sprite_count: int) -> tuple[int, int, list[int]]:
    cols = random.sample(COLOR_POOL, sprite_count + 2)
    bg = cols[0]
    corner_color = cols[1]
    return bg, corner_color, cols[2:]


def _can_place(local: np.ndarray, coords: list[tuple[int, int]], bg: int) -> bool:
    h, w = local.shape
    cells = set(coords)
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(local[r, c]) != bg:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and int(local[rr, cc]) != bg and (rr, cc) not in cells:
                    return False
    return True


def _paint(local: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        local[int(r), int(c)] = int(color)


def _target_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in np.argwhere(mask != 0)]


def _candidate_targets(branch: str) -> list[tuple[np.ndarray, int, int]]:
    row_range = range(4, 10)
    col_range = range(4, 10 if branch == "four" else 8)
    out = []
    for base_mask in TARGET_MASKS:
        for mask in _shape_variants(base_mask):
            h, w = mask.shape
            for top in row_range:
                for left in col_range:
                    if top + h > 10:
                        continue
                    if left + w > (10 if branch == "four" else 8):
                        continue
                    out.append((mask, int(top), int(left)))
    random.shuffle(out)
    return out


def _build_local(branch: str) -> np.ndarray:
    sprite_count = 3 if branch == "four" else 4
    bg, corner_color, sprite_colors = _palette(sprite_count)
    local = np.full(BASE_SHAPE, bg, dtype=int)

    for r, c in FOUR_CORNERS:
        local[r, c] = corner_color
    if branch == "six":
        for r, c in SIX_EXTRA_CORNERS:
            local[r, c] = corner_color

    _paint(local, FILLER_A, sprite_colors[1])
    _paint(local, FILLER_B, sprite_colors[2])

    target_candidates = _candidate_targets(branch)
    six_fillers = SIX_FILLERS.copy()
    random.shuffle(six_fillers)

    for mask, top, left in target_candidates:
        work = local.copy()
        target = _target_coords(mask, top, left)
        if not _can_place(work, target, bg):
            continue
        _paint(work, target, sprite_colors[0])
        if branch == "four":
            if _semantic_ok(work):
                return work
            continue
        for extra in six_fillers:
            attempt = work.copy()
            if not _can_place(attempt, extra, bg):
                continue
            _paint(attempt, extra, sprite_colors[3])
            if _semantic_ok(attempt):
                return attempt

    raise AssertionError("failed to build a valid local pattern")


def _random_canvas(local: np.ndarray, bg: int) -> np.ndarray:
    rot_k = random.randrange(4)
    flip_lr = bool(random.getrandbits(1))
    local = _transform_mask(local, rot_k, flip_lr)
    h, w = local.shape

    top_pad = random.randint(1, 30 - h - 1)
    bottom_pad = random.randint(1, 30 - h - top_pad)
    left_pad = random.randint(1, 30 - w - 1)
    right_pad = random.randint(1, 30 - w - left_pad)

    out = np.full((h + top_pad + bottom_pad, w + left_pad + right_pad), bg, dtype=int)
    out[top_pad : top_pad + h, left_pad : left_pad + w] = local
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        branch = random.choice(["four", "six"])
        local = _build_local(branch)
        grid = _random_canvas(local, int(get_bg(local)))
        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate valid input")
