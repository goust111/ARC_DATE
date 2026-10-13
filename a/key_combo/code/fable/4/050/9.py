from __future__ import annotations

import random
from collections import Counter

import numpy as np


RING_CELLS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
ALL_KEY_CELLS = tuple((r, c) for r in range(3) for c in range(3))
PALETTE = tuple(range(1, 10))
BG = 0
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_is_isolated(grid: np.ndarray, top: int, left: int, frame_color: int) -> bool:
    for dr, dc in RING_CELLS:
        if int(grid[top + dr, left + dc]) != frame_color:
            continue
        rr = top + dr
        cc = left + dc
        for nr, nc in _neighbors4(rr, cc):
            inside = top <= nr < top + 3 and left <= nc < left + 3
            if not inside and _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == frame_color:
                return False
    return True


def _classify_key_window(grid: np.ndarray, bg: int, top: int, left: int) -> dict | None:
    patch = grid[top : top + 3, left : left + 3]
    center = int(patch[1, 1])
    ring = [int(patch[r, c]) for r, c in RING_CELLS]

    if len(set(ring)) == 1:
        frame = ring[0]
        if frame != bg and center != frame and _frame_is_isolated(grid, top, left, frame):
            return {
                "top": top,
                "left": left,
                "frame": frame,
                "center": center,
                "kind": "complete",
            }

    counts = Counter(ring)
    if counts.get(bg, 0) == 1:
        frame_colors = [color for color, count in counts.items() if color != bg and count == 7]
        if len(frame_colors) == 1:
            frame = int(frame_colors[0])
            if center != frame and _frame_is_isolated(grid, top, left, frame):
                return {
                    "top": top,
                    "left": left,
                    "frame": frame,
                    "center": center,
                    "kind": "pseudo",
                }

    return None


def _find_keys(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    keys: list[dict] = []
    for top in range(h - 2):
        for left in range(w - 2):
            if occupied[top : top + 3, left : left + 3].any():
                continue
            key = _classify_key_window(grid, bg, top, left)
            if key is None:
                continue
            occupied[top : top + 3, left : left + 3] = True
            keys.append(key)
    return keys


def _component_holes(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    empty_mask = local != color
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(empty_mask):
        if any(rr == 0 or cc == 0 or rr == local.shape[0] - 1 or cc == local.shape[1] - 1 for rr, cc in comp):
            continue
        holes.append([(r0 + rr, c0 + cc) for rr, cc in comp])
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """Use each complete 3x3 key frame as a color legend, fill matching enclosed holes, then burn every key away.

    Input structure:
    - A background canvas contains large monochrome objects, and some of those objects have enclosed holes.
    - Scattered 3x3 key motifs appear on the background or inside holes: complete keys have an 8-cell monochrome frame and a different-colored center.
    - Distractors include dead complete keys whose frame color matches no holed object, plus pseudo keys with one missing frame cell.

    Rule:
    - Detect every standalone 3x3 key-like motif and erase its whole 3x3 window to the background in the output.
    - For each complete key, use its frame color to find every same-colored object that still has enclosed holes after keys are removed.
    - Fill those hole regions with the key center color; dead keys do nothing, and hole fills can overwrite keys that were hidden inside holes.

    Key insight:
    - The 3x3 motifs are legends rather than scene objects: read complete frames as instructions, then delete all keys before applying any resulting hole fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    keys = _find_keys(g, bg)

    cleaned = g.copy()
    for key in keys:
        top = key["top"]
        left = key["left"]
        cleaned[top : top + 3, left : left + 3] = bg

    fill_by_frame: dict[int, int] = {}
    for key in keys:
        if key["kind"] != "complete":
            continue
        frame = int(key["frame"])
        center = int(key["center"])
        if frame in fill_by_frame:
            assert fill_by_frame[frame] == center, "conflicting keys for the same object color"
        else:
            fill_by_frame[frame] = center

    holes_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    for color in sorted(set(map(int, cleaned.ravel())) - {bg}):
        for comp in _components(cleaned == color):
            holes = _component_holes(cleaned, color, comp)
            if holes:
                holes_by_color.setdefault(int(color), []).extend(holes)

    out = cleaned.copy()
    for frame_color, fill_color in fill_by_frame.items():
        for hole in holes_by_color.get(frame_color, []):
            for r, c in hole:
                out[r, c] = fill_color
    return out


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != BG
    assert not np.any((sub != BG) & mask)
    sub[mask] = patch[mask]


def _can_place_bbox(grid: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = grid.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not np.any(grid[r0:r1, c0:c1] != BG)


def _find_patch_location(grid: np.ndarray, patch: np.ndarray, pad: int) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates = [
        (top, left)
        for top in range(grid.shape[0] - h + 1)
        for left in range(grid.shape[1] - w + 1)
        if _can_place_bbox(grid, top, left, h, w, pad)
    ]
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _make_holed_patch(color: int, require_large_hole: bool) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    for _ in range(200):
        min_side = 9 if require_large_hole else 7
        h = random.randint(min_side, 12)
        w = random.randint(min_side, 12)
        patch = np.full((h, w), color, dtype=int)

        min_hole_h = 5 if require_large_hole else 3
        min_hole_w = 5 if require_large_hole else 3
        max_hole_h = min(6, h - 2)
        max_hole_w = min(6, w - 2)
        if min_hole_h > max_hole_h or min_hole_w > max_hole_w:
            continue

        holes: list[tuple[int, int, int, int]] = []
        hole_h = random.randint(min_hole_h, max_hole_h)
        hole_w = random.randint(min_hole_w, max_hole_w)
        top = random.randint(1, h - hole_h - 1)
        left = random.randint(1, w - hole_w - 1)
        patch[top : top + hole_h, left : left + hole_w] = BG
        holes.append((top, left, top + hole_h - 1, left + hole_w - 1))

        if random.random() < 0.65:
            for _ in range(80):
                hole2_h = random.randint(2, min(4, h - 2))
                hole2_w = random.randint(2, min(4, w - 2))
                top2 = random.randint(1, h - hole2_h - 1)
                left2 = random.randint(1, w - hole2_w - 1)
                row_gap = top + hole_h < top2 or top2 + hole2_h < top
                col_gap = left + hole_w < left2 or left2 + hole2_w < left
                if not (row_gap or col_gap):
                    continue
                if not (
                    top2 + hole2_h <= top - 1
                    or top + hole_h <= top2 - 1
                    or left2 + hole2_w <= left - 1
                    or left + hole_w <= left2 - 1
                ):
                    continue
                patch[top2 : top2 + hole2_h, left2 : left2 + hole2_w] = BG
                holes.append((top2, left2, top2 + hole2_h - 1, left2 + hole2_w - 1))
                break

        return patch, holes
    raise RuntimeError("failed to build a holed patch")


def _make_solid_patch(color: int) -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    return np.full((h, w), color, dtype=int)


def _reserve_is_free(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not reserved[r0:r1, c0:c1].any()


def _mark_reserved(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    reserved[r0:r1, c0:c1] = True


def _can_place_key_here(grid: np.ndarray, reserved: np.ndarray, top: int, left: int, frame_color: int) -> bool:
    if not _reserve_is_free(reserved, top, left, 3, 3, pad=1):
        return False
    if np.any(grid[top : top + 3, left : left + 3] != BG):
        return False
    for dr, dc in RING_CELLS:
        rr = top + dr
        cc = left + dc
        for nr, nc in _neighbors4(rr, cc):
            inside = top <= nr < top + 3 and left <= nc < left + 3
            if not inside and _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == frame_color:
                return False
    return True


def _place_key_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    frame_color: int,
    center_color: int,
    kind: str,
    hole_box: tuple[int, int, int, int] | None = None,
) -> tuple[int, int] | None:
    patch = np.full((3, 3), frame_color, dtype=int)
    patch[1, 1] = center_color
    if kind == "pseudo":
        miss_r, miss_c = random.choice(RING_CELLS)
        patch[miss_r, miss_c] = BG

    candidates: list[tuple[int, int]] = []
    if hole_box is None:
        for top in range(grid.shape[0] - 2):
            for left in range(grid.shape[1] - 2):
                if _can_place_key_here(grid, reserved, top, left, frame_color):
                    candidates.append((top, left))
    else:
        r0, c0, r1, c1 = hole_box
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            return None
        for top in range(r0 + 1, r1 - 2):
            for left in range(c0 + 1, c1 - 2):
                if _can_place_key_here(grid, reserved, top, left, frame_color):
                    candidates.append((top, left))

    random.shuffle(candidates)
    if not candidates:
        return None
    top, left = candidates[0]
    _paste_patch(grid, patch, top, left)
    _mark_reserved(reserved, top, left, 3, 3, pad=1)
    return top, left


def _random_center_color(frame_color: int) -> int:
    choices = [color for color in PALETTE if color != frame_color]
    return random.choice(choices)


def _build_expected(grid: np.ndarray, objects: list[dict], keys: list[dict]) -> np.ndarray:
    out = grid.copy()
    valid_fill = {key["frame"]: key["center"] for key in keys if key["kind"] == "complete" and key["valid"]}
    for key in keys:
        top = key["top"]
        left = key["left"]
        out[top : top + 3, left : left + 3] = BG
    for obj in objects:
        fill_color = valid_fill.get(obj["color"])
        if fill_color is None:
            continue
        for r0, c0, r1, c1 in obj["holes"]:
            out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    return out


def _generate_case() -> np.ndarray | None:
    h = random.randint(19, 28)
    w = random.randint(19, 28)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    n_holed = random.choice([1, 2])
    n_solid = random.choice([0, 1, 2])
    needed_colors = n_holed + n_solid
    object_colors = random.sample(PALETTE, needed_colors)
    holed_colors = object_colors[:n_holed]
    solid_colors = object_colors[n_holed:]

    objects: list[dict] = []
    holed_objects: list[dict] = []

    for idx, color in enumerate(holed_colors):
        patch, hole_boxes = _make_holed_patch(color, require_large_hole=(idx == 0))
        pos = _find_patch_location(grid, patch, pad=2)
        if pos is None:
            return None
        top, left = pos
        _paste_patch(grid, patch, top, left)
        abs_holes = [(top + r0, left + c0, top + r1, left + c1) for r0, c0, r1, c1 in hole_boxes]
        obj = {"color": color, "holes": abs_holes}
        objects.append(obj)
        holed_objects.append(obj)

    for color in solid_colors:
        patch = _make_solid_patch(color)
        pos = _find_patch_location(grid, patch, pad=2)
        if pos is None:
            return None
        _paste_patch(grid, patch, pos[0], pos[1])
        objects.append({"color": color, "holes": []})

    keys: list[dict] = []
    inside_hole_used = False
    for idx, obj in enumerate(holed_objects):
        must_place = idx == 0
        if not must_place and random.random() < 0.4:
            continue
        center = _random_center_color(obj["color"])
        hole_target = None
        if not inside_hole_used:
            large_holes = [box for box in obj["holes"] if box[2] - box[0] + 1 >= 5 and box[3] - box[1] + 1 >= 5]
            if large_holes and random.random() < 0.7:
                hole_target = random.choice(large_holes)
        placed = _place_key_patch(grid, reserved, obj["color"], center, kind="complete", hole_box=hole_target)
        if placed is None:
            placed = _place_key_patch(grid, reserved, obj["color"], center, kind="complete", hole_box=None)
            hole_target = None
        if placed is None:
            return None
        inside_hole_used = inside_hole_used or hole_target is not None
        keys.append(
            {
                "top": placed[0],
                "left": placed[1],
                "frame": obj["color"],
                "center": center,
                "kind": "complete",
                "valid": True,
            }
        )

    dead_frame_choices = [color for color in PALETTE if color not in holed_colors]
    if not dead_frame_choices:
        return None
    dead_frame = random.choice(dead_frame_choices)
    dead_center = _random_center_color(dead_frame)
    dead_pos = _place_key_patch(grid, reserved, dead_frame, dead_center, kind="complete", hole_box=None)
    if dead_pos is None:
        return None
    keys.append(
        {
            "top": dead_pos[0],
            "left": dead_pos[1],
            "frame": dead_frame,
            "center": dead_center,
            "kind": "complete",
            "valid": False,
        }
    )

    pseudo_frame = random.choice(PALETTE)
    pseudo_center = _random_center_color(pseudo_frame)
    pseudo_pos = _place_key_patch(grid, reserved, pseudo_frame, pseudo_center, kind="pseudo", hole_box=None)
    if pseudo_pos is None:
        return None
    keys.append(
        {
            "top": pseudo_pos[0],
            "left": pseudo_pos[1],
            "frame": pseudo_frame,
            "center": pseudo_center,
            "kind": "pseudo",
            "valid": False,
        }
    )

    expected = _build_expected(grid, objects, keys)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if len(_find_keys(grid, BG)) != len(keys):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-legend puzzle")
