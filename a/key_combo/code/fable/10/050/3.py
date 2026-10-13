from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(pts, dtype=int)
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = patch == color
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                }
            )
    return components


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(v) for v in coords[0])
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _half_counts(mask: np.ndarray, axis: str) -> tuple[int, int]:
    h, w = mask.shape
    if axis == "lr":
        left = int(mask[:, : w // 2].sum())
        right = int(mask[:, w // 2 + (w % 2) :].sum())
        return left, right
    if axis == "tb":
        top = int(mask[: h // 2, :].sum())
        bottom = int(mask[h // 2 + (h % 2) :, :].sum())
        return top, bottom
    raise ValueError(f"unknown axis: {axis}")


def _analyze_mask(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise AssertionError("sprite mask must be non-empty 2D")
    if not _is_connected(mask):
        raise AssertionError("sprite must be 4-connected")

    h_sym = bool(np.array_equal(mask, mask[::-1, :]))
    v_sym = bool(np.array_equal(mask, mask[:, ::-1]))
    if h_sym == v_sym:
        raise AssertionError("sprite must have exactly one mirror axis")

    if h_sym:
        left, right = _half_counts(mask, "lr")
        if left == right:
            raise AssertionError("left/right mass must be imbalanced")
        step = abs(left - right)
        return {
            "symmetry": "horizontal",
            "move": (0, 1 if right > left else -1),
            "step": step,
            "counts": (left, right),
        }

    top, bottom = _half_counts(mask, "tb")
    if top == bottom:
        raise AssertionError("top/bottom mass must be imbalanced")
    step = abs(top - bottom)
    return {
        "symmetry": "vertical",
        "move": (1 if bottom > top else -1, 0),
        "step": step,
        "counts": (top, bottom),
    }


def _paint_mask(dst: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    dst[top : top + h, left : left + w][mask] = int(color)


def _translate_mask(mask: np.ndarray, top: int, left: int, dr: int, dc: int) -> np.ndarray:
    coords = np.argwhere(mask)
    out = coords.copy()
    out[:, 0] += int(top + dr)
    out[:, 1] += int(left + dc)
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    colors = [comp["color"] for comp in components]
    if len(colors) != len(set(colors)):
        raise AssertionError("each sprite must use a distinct color")

    out = np.full_like(g, bg)
    occupied = np.zeros(g.shape, dtype=bool)
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        meta = _analyze_mask(comp["mask"])
        dr = meta["move"][0] * meta["step"]
        dc = meta["move"][1] * meta["step"]
        target = _translate_mask(comp["mask"], r0, c0, dr, dc)
        if (
            target[:, 0].min() < 0
            or target[:, 1].min() < 0
            or target[:, 0].max() >= g.shape[0]
            or target[:, 1].max() >= g.shape[1]
        ):
            raise AssertionError("translated sprite leaves the grid")
        if np.any(occupied[target[:, 0], target[:, 1]]):
            raise AssertionError("translated sprites overlap")
        occupied[target[:, 0], target[:, 1]] = True
        out[target[:, 0], target[:, 1]] = comp["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each sprite along the axis selected by its unique mirror symmetry, using side-mass imbalance as signed distance.

    Input structure:
    - A black background contains several separated monochrome sprites of distinct colors.
    - Each sprite is 4-connected and is symmetric about exactly one mid-axis of its own bounding box:
      either the horizontal axis or the vertical axis.
    - The other axis is not symmetric, so its two halves have different pixel counts.

    Rule:
    - If a sprite is symmetric about its horizontal mid-axis, move it horizontally.
    - If a sprite is symmetric about its vertical mid-axis, move it vertically.
    - The heavier half on the non-symmetric axis gives the direction, and the absolute pixel-count
      difference between those two halves gives the number of cells to translate.
    - Preserve every sprite's shape and color; only its position changes.

    Key insight:
    - The sprite itself encodes all motion data at once: one symmetry axis chooses which coordinate
      may change, while the imbalance across the other axis provides both the sign and the magnitude.
    """
    return _expected_output(np.asarray(grid, dtype=int))


def _mask_from_column_profile(height: int, profile: list[int]) -> np.ndarray:
    width = len(profile)
    mask = np.zeros((height, width), dtype=bool)
    mid = height // 2
    for c, filled in enumerate(profile):
        if height % 2 == 1:
            reach = (filled - 1) // 2
            r0 = mid - reach
            r1 = mid + reach
        else:
            reach = (filled - 2) // 2
            r0 = mid - 1 - reach
            r1 = mid + reach
        mask[r0 : r1 + 1, c] = True
    return mask


def _mask_from_row_profile(width: int, profile: list[int]) -> np.ndarray:
    height = len(profile)
    mask = np.zeros((height, width), dtype=bool)
    mid = width // 2
    for r, filled in enumerate(profile):
        if width % 2 == 1:
            reach = (filled - 1) // 2
            c0 = mid - reach
            c1 = mid + reach
        else:
            reach = (filled - 2) // 2
            c0 = mid - 1 - reach
            c1 = mid + reach
        mask[r, c0 : c1 + 1] = True
    return mask


def _sample_horizontal_sprite(rng: random.Random) -> tuple[np.ndarray, dict]:
    for _ in range(400):
        h = rng.randint(3, 7)
        w = rng.randint(4, 7)
        allowed = list(range(1, h + 1, 2)) if h % 2 == 1 else list(range(2, h + 1, 2))
        profile = [rng.choice(allowed) for _ in range(w)]
        if profile == profile[::-1]:
            continue
        if len(set(profile)) == 1:
            continue
        mask = _mask_from_column_profile(h, profile)
        area = int(mask.sum())
        if not (6 <= area <= 20):
            continue
        fill = area / float(h * w)
        if not (0.35 <= fill <= 0.9):
            continue
        try:
            meta = _analyze_mask(mask)
        except AssertionError:
            continue
        if meta["symmetry"] != "horizontal" or not (1 <= meta["step"] <= 4):
            continue
        return mask, meta
    raise RuntimeError("failed to sample a horizontal-symmetry sprite")


def _sample_vertical_sprite(rng: random.Random) -> tuple[np.ndarray, dict]:
    for _ in range(400):
        h = rng.randint(4, 7)
        w = rng.randint(3, 7)
        allowed = list(range(1, w + 1, 2)) if w % 2 == 1 else list(range(2, w + 1, 2))
        profile = [rng.choice(allowed) for _ in range(h)]
        if profile == profile[::-1]:
            continue
        if len(set(profile)) == 1:
            continue
        mask = _mask_from_row_profile(w, profile)
        area = int(mask.sum())
        if not (6 <= area <= 20):
            continue
        fill = area / float(h * w)
        if not (0.35 <= fill <= 0.9):
            continue
        try:
            meta = _analyze_mask(mask)
        except AssertionError:
            continue
        if meta["symmetry"] != "vertical" or not (1 <= meta["step"] <= 4):
            continue
        return mask, meta
    raise RuntimeError("failed to sample a vertical-symmetry sprite")


def _build_slots(rng: random.Random) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    top_h = rng.randint(10, 13)
    bottom_h = rng.randint(10, 13)
    left_w = rng.randint(10, 13)
    right_w = rng.randint(10, 13)
    gap_r = rng.randint(2, 4)
    gap_c = rng.randint(2, 4)

    rows = top_h + gap_r + bottom_h
    cols = left_w + gap_c + right_w
    slots = [
        (0, 0, top_h - 1, left_w - 1),
        (0, left_w + gap_c, top_h - 1, cols - 1),
        (top_h + gap_r, 0, rows - 1, left_w - 1),
        (top_h + gap_r, left_w + gap_c, rows - 1, cols - 1),
    ]
    return rows, cols, slots


def _sample_top_left(
    rng: random.Random,
    slot: tuple[int, int, int, int],
    mask: np.ndarray,
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = slot
    h, w = mask.shape
    margin = 1

    row_min = r0 + margin + max(0, -dr)
    row_max = r1 - h - margin + 1 - max(0, dr)
    col_min = c0 + margin + max(0, -dc)
    col_max = c1 - w - margin + 1 - max(0, dc)
    if row_min > row_max or col_min > col_max:
        return None
    return rng.randint(row_min, row_max), rng.randint(col_min, col_max)


def _sweep_cells(mask: np.ndarray, top: int, left: int, dr: int, dc: int) -> set[tuple[int, int]]:
    coords = np.argwhere(mask)
    steps = max(abs(dr), abs(dc))
    out: set[tuple[int, int]] = set()
    step_r = 0 if dr == 0 else dr // abs(dr)
    step_c = 0 if dc == 0 else dc // abs(dc)
    for k in range(steps + 1):
        rr = top + k * step_r
        cc = left + k * step_c
        for r, c in coords:
            out.add((int(rr + r), int(cc + c)))
    return out


def _instance_metadata(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    components = _find_components(grid, bg)
    metas = []
    for comp in components:
        meta = _analyze_mask(comp["mask"])
        meta = dict(meta)
        meta["color"] = comp["color"]
        meta["bbox"] = comp["bbox"]
        meta["mask"] = comp["mask"]
        metas.append(meta)
    return metas


def _validate_semantics(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    bg = _get_bg(g)
    assert bg == 0

    components = _find_components(g, bg)
    assert 2 <= len(components) <= 4
    assert len({comp["color"] for comp in components}) == len(components)

    axis_kinds = set()
    signs = set()
    large_step = False
    swept_union: set[tuple[int, int]] = set()

    for comp in components:
        meta = _analyze_mask(comp["mask"])
        axis_kinds.add(meta["symmetry"])
        dr = meta["move"][0] * meta["step"]
        dc = meta["move"][1] * meta["step"]
        signs.add(int(np.sign(dr + dc)))
        large_step |= meta["step"] >= 2
        r0, c0, _r1, _c1 = comp["bbox"]
        swept = _sweep_cells(comp["mask"], r0, c0, dr, dc)
        assert swept.isdisjoint(swept_union)
        swept_union |= swept

    assert axis_kinds == {"horizontal", "vertical"}
    assert signs == {-1, 1}
    assert large_step

    expected = _expected_output(g)
    actual = main(g)
    assert np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(500):
        rows, cols, slots = _build_slots(rng)
        grid = np.zeros((rows, cols), dtype=int)

        count = rng.randint(2, 4)
        slot_ids = rng.sample(range(4), count)
        axis_plan = ["horizontal", "vertical"]
        while len(axis_plan) < count:
            axis_plan.append(rng.choice(["horizontal", "vertical"]))
        rng.shuffle(axis_plan)

        colors = rng.sample(list(range(1, 10)), count)

        for slot_id, axis, color in zip(slot_ids, axis_plan, colors):
            if axis == "horizontal":
                mask, meta = _sample_horizontal_sprite(rng)
            else:
                mask, meta = _sample_vertical_sprite(rng)

            dr = meta["move"][0] * meta["step"]
            dc = meta["move"][1] * meta["step"]
            top_left = _sample_top_left(rng, slots[slot_id], mask, dr, dc)
            if top_left is None:
                break
            _paint_mask(grid, mask, top_left[0], top_left[1], color)
        else:
            try:
                _validate_semantics(grid)
                return grid
            except AssertionError:
                pass

    raise RuntimeError("failed to generate a valid input for puzzle 3")
