from __future__ import annotations

import random
from math import ceil

import numpy as np


BG_COLOR = 0
ARROW_COLOR = 2
DIVIDER_COLOR = 5
STAMP_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400

STAMP_MASKS = (
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
)

ARROW_PATTERNS = {
    "right": (
        ((0, 0), (0, 1), (1, 1)),
        ((1, 0), (0, 1), (1, 1)),
    ),
    "left": (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
    ),
    "down": (
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
    ),
    "up": (
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (0, 1), (1, 1)),
    ),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, a[0] - b[2] - 1, b[0] - a[2] - 1)
    col_gap = max(0, a[1] - b[3] - 1, b[1] - a[3] - 1)
    return int(row_gap + col_gap)


def _find_divider(grid: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = grid.shape
    full_rows = [r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])]
    full_cols = [c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]
    assert len(full_rows) + len(full_cols) == 1, "expected one full divider line"
    if full_rows:
        return "h", int(full_rows[0])
    return "v", int(full_cols[0])


def _component_side(bbox: tuple[int, int, int, int], divider_axis: str, divider_index: int) -> int:
    if divider_axis == "v":
        center = (bbox[1] + bbox[3]) / 2.0
    else:
        center = (bbox[0] + bbox[2]) / 2.0
    return -1 if center < divider_index else 1


def _is_arrow_component(grid: np.ndarray, coords: np.ndarray) -> bool:
    if coords.shape[0] != 3:
        return False
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    return (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _paste_non_bg(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    if not np.any(mask):
        return
    target = out[top : top + patch.shape[0], left : left + patch.shape[1]]
    target[mask] = patch[mask]


def _mirror_bbox(
    bbox: tuple[int, int, int, int], divider_axis: str, divider_index: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    if divider_axis == "v":
        return (r0, 2 * divider_index - c1, r1, 2 * divider_index - c0)
    return (2 * divider_index - r1, c0, 2 * divider_index - r0, c1)


def _direction_from_pair(
    arrow_bbox: tuple[int, int, int, int], stamp_bbox: tuple[int, int, int, int]
) -> str:
    if arrow_bbox[3] < stamp_bbox[1]:
        return "right"
    if arrow_bbox[1] > stamp_bbox[3]:
        return "left"
    if arrow_bbox[2] < stamp_bbox[0]:
        return "down"
    if arrow_bbox[0] > stamp_bbox[2]:
        return "up"

    arrow_center = ((arrow_bbox[0] + arrow_bbox[2]) / 2.0, (arrow_bbox[1] + arrow_bbox[3]) / 2.0)
    stamp_center = ((stamp_bbox[0] + stamp_bbox[2]) / 2.0, (stamp_bbox[1] + stamp_bbox[3]) / 2.0)
    dr = stamp_center[0] - arrow_center[0]
    dc = stamp_center[1] - arrow_center[1]
    if abs(dc) >= abs(dr):
        return "right" if dc > 0 else "left"
    return "down" if dr > 0 else "up"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-stamp each source motif across the divider after flipping it by the nearby arrow type.

    Input structure:
    - The dominant color is a black background.
    - A single grey full-row or full-column divider splits the grid into two halves.
    - One half contains several small multicolor stamp components and, beside each one,
      a separate red three-pixel L-shaped arrow marker.
    - The opposite half is empty background and serves as the stamp target area.

    Rule:
    - Identify each red L marker and pair it with the nearest non-arrow component on the same side of the divider.
    - The arrow selects the flip axis: left/right arrows mean a horizontal mirror,
      while up/down arrows mean a vertical mirror.
    - Reflect the stamp's bounding box across the divider, flip the stamp inside that box,
      and draw the flipped stamp on the mirrored target box.
    - Keep the original stamp and the divider, but erase every arrow back to background.

    Key insight:
    - The arrow does not tell the stamp where to move; it tells the stamp how to flip,
      while the final destination is set only by mirror symmetry across the divider.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    divider_axis, divider_index = _find_divider(g, bg)

    fg = g != bg
    if divider_axis == "v":
        fg[:, divider_index] = False
    else:
        fg[divider_index, :] = False

    arrow_infos = []
    stamp_infos = []
    for coords in _components(fg):
        bbox = _bbox(coords)
        info = {"coords": coords, "bbox": bbox, "side": _component_side(bbox, divider_axis, divider_index)}
        if _is_arrow_component(g, coords):
            arrow_infos.append(info)
        else:
            stamp_infos.append(info)

    assert arrow_infos and stamp_infos and len(arrow_infos) <= len(stamp_infos)

    unused_stamps = list(range(len(stamp_infos)))
    pairs = []
    for arrow in sorted(arrow_infos, key=lambda item: item["bbox"]):
        candidates = [
            idx
            for idx in unused_stamps
            if stamp_infos[idx]["side"] == arrow["side"]
        ]
        assert candidates, "every arrow must have a stamp on the same side"
        best_idx = min(
            candidates,
            key=lambda idx: (
                _bbox_distance(arrow["bbox"], stamp_infos[idx]["bbox"]),
                stamp_infos[idx]["bbox"],
            ),
        )
        unused_stamps.remove(best_idx)
        pairs.append((arrow, stamp_infos[best_idx]))

    out = g.copy()
    for arrow, _stamp in pairs:
        out[arrow["coords"][:, 0], arrow["coords"][:, 1]] = bg

    for arrow, stamp in sorted(pairs, key=lambda pair: pair[1]["bbox"]):
        patch = _extract_patch(g, stamp["coords"], bg)
        direction = _direction_from_pair(arrow["bbox"], stamp["bbox"])
        if direction in ("left", "right"):
            patch = np.fliplr(patch)
        else:
            patch = np.flipud(patch)
        target_bbox = _mirror_bbox(stamp["bbox"], divider_axis, divider_index)
        r0, c0, _r1, _c1 = target_bbox
        _paste_non_bg(out, patch, r0, c0, bg)

    return out


def _random_mask(rng: random.Random) -> np.ndarray:
    mask = np.array(rng.choice(STAMP_MASKS), copy=True)
    for _ in range(rng.randrange(4)):
        mask = np.rot90(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _random_stamp_patch(direction: str, rng: random.Random) -> np.ndarray:
    mask = _random_mask(rng)
    cells = [tuple(map(int, pt)) for pt in np.argwhere(mask == 1)]
    n_colors = rng.randint(2, min(4, len(cells)))
    palette = rng.sample(STAMP_COLORS, n_colors)

    for _ in range(80):
        patch = np.full(mask.shape, BG_COLOR, dtype=int)
        picks = [rng.choice(palette) for _ in cells]
        if len(set(picks)) < 2:
            continue
        for (r, c), color in zip(cells, picks):
            patch[r, c] = color
        flipped = np.fliplr(patch) if direction in ("left", "right") else np.flipud(patch)
        if not np.array_equal(patch, flipped):
            return patch

    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    for idx, (r, c) in enumerate(cells):
        patch[r, c] = palette[idx % len(palette)]
    flipped = np.fliplr(patch) if direction in ("left", "right") else np.flipud(patch)
    if np.array_equal(patch, flipped):
        first_r, first_c = cells[0]
        last_r, last_c = cells[-1]
        patch[first_r, first_c] = palette[0]
        patch[last_r, last_c] = palette[-1]
    return patch


def _make_object(direction: str, rng: random.Random) -> dict:
    patch = _random_stamp_patch(direction, rng)
    stamp_h, stamp_w = patch.shape
    pattern = rng.choice(ARROW_PATTERNS[direction])

    if direction == "right":
        arrow_top = rng.randint(0, stamp_h - 2)
        stamp_tl = (0, 3)
        arrow_tl = (arrow_top, 0)
        local_h, local_w = stamp_h, stamp_w + 3
    elif direction == "left":
        arrow_top = rng.randint(0, stamp_h - 2)
        stamp_tl = (0, 0)
        arrow_tl = (arrow_top, stamp_w + 1)
        local_h, local_w = stamp_h, stamp_w + 3
    elif direction == "down":
        arrow_left = rng.randint(0, stamp_w - 2)
        stamp_tl = (3, 0)
        arrow_tl = (0, arrow_left)
        local_h, local_w = stamp_h + 3, stamp_w
    else:
        arrow_left = rng.randint(0, stamp_w - 2)
        stamp_tl = (0, 0)
        arrow_tl = (stamp_h + 1, arrow_left)
        local_h, local_w = stamp_h + 3, stamp_w

    arrow_cells = [(arrow_tl[0] + dr, arrow_tl[1] + dc) for dr, dc in pattern]
    return {
        "direction": direction,
        "patch": patch,
        "local_h": int(local_h),
        "local_w": int(local_w),
        "stamp_tl": stamp_tl,
        "arrow_cells": arrow_cells,
    }


def _mark_rect(occ: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occ[r0 : r1 + 1, c0 : c1 + 1] = True


def _rect_is_free(occ: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occ.shape[0] - 1, r1 + 1)
    cc1 = min(occ.shape[1] - 1, c1 + 1)
    return not bool(np.any(occ[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _choose_canvas(objects: list[dict], orientation: str, rng: random.Random) -> tuple[int, int] | None:
    total_area = sum(obj["local_h"] * obj["local_w"] for obj in objects)
    max_h = max(obj["local_h"] for obj in objects)
    max_w = max(obj["local_w"] for obj in objects)

    if orientation == "v":
        min_half_w = max(9, max_w + 2)
        if min_half_w > 14:
            return None
        half_w = rng.randint(min_half_w, 14)
        usable_w = half_w - 2
        min_h = max(12, max_h + 2, int(ceil(total_area / max(1, usable_w))) + 4)
        if min_h > 29:
            return None
        height = rng.randint(min_h, 29)
        width = 2 * half_w + 1
        return (height, width)

    min_half_h = max(9, max_h + 2)
    if min_half_h > 14:
        return None
    half_h = rng.randint(min_half_h, 14)
    usable_h = half_h - 2
    min_w = max(12, max_w + 2, int(ceil(total_area / max(1, usable_h))) + 4)
    if min_w > 29:
        return None
    width = rng.randint(min_w, 29)
    height = 2 * half_h + 1
    return (height, width)


def _place_objects(
    objects: list[dict], orientation: str, source_side: str, rng: random.Random
) -> tuple[np.ndarray, list[dict]] | None:
    canvas = _choose_canvas(objects, orientation, rng)
    if canvas is None:
        return None
    height, width = canvas
    grid = np.full((height, width), BG_COLOR, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    if orientation == "v":
        divider = width // 2
        grid[:, divider] = DIVIDER_COLOR
        if source_side == "left":
            src_r0, src_c0, src_r1, src_c1 = 0, 0, height - 1, divider - 1
        else:
            src_r0, src_c0, src_r1, src_c1 = 0, divider + 1, height - 1, width - 1
    else:
        divider = height // 2
        grid[divider, :] = DIVIDER_COLOR
        if source_side == "top":
            src_r0, src_c0, src_r1, src_c1 = 0, 0, divider - 1, width - 1
        else:
            src_r0, src_c0, src_r1, src_c1 = divider + 1, 0, height - 1, width - 1

    placed: list[dict] = []
    for obj in sorted(objects, key=lambda item: item["local_h"] * item["local_w"], reverse=True):
        local_h = obj["local_h"]
        local_w = obj["local_w"]
        row_lo = src_r0 + 1
        row_hi = src_r1 - local_h
        col_lo = src_c0 + 1
        col_hi = src_c1 - local_w
        if row_lo > row_hi or col_lo > col_hi:
            return None

        positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        rng.shuffle(positions)

        chosen = None
        for top, left in positions:
            bbox = (top, left, top + local_h - 1, left + local_w - 1)
            if _rect_is_free(occ, bbox):
                chosen = (top, left)
                _mark_rect(occ, bbox)
                break
        if chosen is None:
            return None

        top, left = chosen
        stamp_top = top + obj["stamp_tl"][0]
        stamp_left = left + obj["stamp_tl"][1]
        stamp_patch = obj["patch"]
        _paste_non_bg(grid, stamp_patch, stamp_top, stamp_left, BG_COLOR)

        arrow_coords = []
        for dr, dc in obj["arrow_cells"]:
            rr, cc = top + dr, left + dc
            grid[rr, cc] = ARROW_COLOR
            arrow_coords.append((rr, cc))

        placed.append(
            {
                "direction": obj["direction"],
                "patch": stamp_patch,
                "stamp_bbox": (
                    stamp_top,
                    stamp_left,
                    stamp_top + stamp_patch.shape[0] - 1,
                    stamp_left + stamp_patch.shape[1] - 1,
                ),
                "arrow_coords": arrow_coords,
            }
        )

    return grid, placed


def _expected_output(
    grid: np.ndarray, placed: list[dict], orientation: str
) -> np.ndarray:
    out = np.array(grid, copy=True)
    divider_axis, divider_index = ("v", grid.shape[1] // 2) if orientation == "v" else ("h", grid.shape[0] // 2)

    for obj in placed:
        for r, c in obj["arrow_coords"]:
            out[r, c] = BG_COLOR

    for obj in sorted(placed, key=lambda item: item["stamp_bbox"]):
        patch = np.array(obj["patch"], copy=True)
        if obj["direction"] in ("left", "right"):
            patch = np.fliplr(patch)
        else:
            patch = np.flipud(patch)
        target_bbox = _mirror_bbox(obj["stamp_bbox"], divider_axis, divider_index)
        _paste_non_bg(out, patch, target_bbox[0], target_bbox[1], BG_COLOR)

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(2, 4)
    directions = rng.sample(["left", "right", "up", "down"], n_objects)
    objects = [_make_object(direction, rng) for direction in directions]

    orientation = rng.choice(["v", "h"])
    if orientation == "v":
        source_side = rng.choice(["left", "right"])
    else:
        source_side = rng.choice(["top", "bottom"])

    placed_result = _place_objects(objects, orientation, source_side, rng)
    if placed_result is None:
        return None
    grid, placed = placed_result

    expected = _expected_output(grid, placed, orientation)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 15/2")
