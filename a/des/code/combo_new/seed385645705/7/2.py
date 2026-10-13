from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
N_COLORS = 10
GEN_RETRIES = 500
PLACE_RETRIES = 120


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray, bg: int) -> dict[int, list[dict]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    by_color: dict[int, list[dict]] = {}
    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == bg or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((int(r), int(c)))
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            by_color.setdefault(color, []).append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return by_color


def _row_to_cols(coords: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for r, c in coords:
        out.setdefault(r, []).append(c)
    for cols in out.values():
        cols.sort()
    return out


def _col_to_rows(coords: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for r, c in coords:
        out.setdefault(c, []).append(r)
    for rows in out.values():
        rows.sort()
    return out


def _horizontal_candidate(left: dict, right: dict) -> tuple[int, tuple[int, int]] | None:
    if int(left["bbox"][3]) >= int(right["bbox"][1]):
        return None
    left_rows = _row_to_cols(left["coords"])
    right_rows = _row_to_cols(right["coords"])
    common = set(left_rows) & set(right_rows)
    if not common:
        return None
    min_dist = min(
        rc - lc
        for r in common
        for lc in left_rows[r]
        for rc in right_rows[r]
        if rc > lc
    )
    assert min_dist >= 2
    return int(min_dist), (0, -(int(min_dist) - 1))


def _vertical_candidate(top: dict, bottom: dict) -> tuple[int, tuple[int, int]] | None:
    if int(top["bbox"][2]) >= int(bottom["bbox"][0]):
        return None
    top_cols = _col_to_rows(top["coords"])
    bottom_cols = _col_to_rows(bottom["coords"])
    common = set(top_cols) & set(bottom_cols)
    if not common:
        return None
    min_dist = min(
        br - tr
        for c in common
        for tr in top_cols[c]
        for br in bottom_cols[c]
        if br > tr
    )
    assert min_dist >= 2
    return int(min_dist), (-(int(min_dist) - 1), 0)


def _sprite_from_coords(
    coords: list[tuple[int, int]], color: int, bg: int
) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        sprite[r - r0, c - c0] = color
    return sprite


def _merged_sprite(comp_a: dict, comp_b: dict, bg: int) -> dict:
    candidates: list[tuple[str, int, dict, dict, tuple[int, int]]] = []
    for left, right in ((comp_a, comp_b), (comp_b, comp_a)):
        result = _horizontal_candidate(left, right)
        if result is not None:
            dist, shift = result
            candidates.append(("h", dist, left, right, shift))
    for top, bottom in ((comp_a, comp_b), (comp_b, comp_a)):
        result = _vertical_candidate(top, bottom)
        if result is not None:
            dist, shift = result
            candidates.append(("v", dist, top, bottom, shift))

    assert len(candidates) == 1, "same-color fragments must be aligned in exactly one direction"
    _axis, _dist, fixed, moving, (dr, dc) = candidates[0]

    merged = [(int(r), int(c)) for r, c in fixed["coords"]]
    merged.extend((int(r) + dr, int(c) + dc) for r, c in moving["coords"])
    assert len(set(merged)) == len(merged), "translated fragments must not overlap"

    color = int(fixed["color"])
    anchor_bbox = min(comp_a["bbox"], comp_b["bbox"])
    sprite = _sprite_from_coords(merged, color, bg)
    return {
        "area": int(comp_a["area"]) + int(comp_b["area"]),
        "anchor": anchor_bbox[:2],
        "sprite": sprite,
    }


def _paste(out: np.ndarray, sprite: np.ndarray, row: int, col: int, bg: int) -> None:
    mask = sprite != bg
    window = out[row : row + sprite.shape[0], col : col + sprite.shape[1]]
    window[mask] = sprite[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Dock same-color fragments, then pack the assembled shapes by area.

    Input structure:
    - The dominant color is the background.
    - Every foreground color appears as exactly two separated monochrome
      4-connected fragments.
    - The two fragments of a color are aligned left-right or top-bottom, with
      actual same-row or same-column pixels that can meet by sliding.

    Rule:
    - For each color, identify its two 4-connected fragments.
    - Slide the pair toward each other along their shared horizontal or vertical
      direction until some pixels are exactly 4-adjacent, without overlap.
    - Convert each docked pair to its tight bounding-box sprite and compute its
      pixel area.
    - Sort docked sprites by area from small to large, then paste them left to
      right in the smallest background output grid, with one background column
      between neighbors and all sprites bottom-aligned.

    Key insight:
    The fragments are not independent objects: same-color pieces must first be
    docked into larger shapes, and those assembled shapes are then reordered by
    area instead of by their input positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    assembled: list[dict] = []
    for color, comps in _components_by_color(g, bg).items():
        assert len(comps) == 2, f"color {color} must have exactly two fragments"
        assembled.append(_merged_sprite(comps[0], comps[1], bg))

    assert assembled, "expected at least one foreground color"
    assembled.sort(
        key=lambda item: (
            int(item["area"]),
            int(item["anchor"][0]),
            int(item["anchor"][1]),
            int(item["sprite"].shape[0]),
            int(item["sprite"].shape[1]),
        )
    )

    out_h = max(int(item["sprite"].shape[0]) for item in assembled)
    out_w = sum(int(item["sprite"].shape[1]) for item in assembled) + len(assembled) - 1
    assert out_h <= 30 and out_w <= 30
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for item in assembled:
        sprite = item["sprite"]
        row = out_h - int(sprite.shape[0])
        _paste(out, sprite, row, col, bg)
        col += int(sprite.shape[1]) + 1
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_fragment() -> np.ndarray:
    for _ in range(50):
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        area = random.randint(1, min(7, h * w))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == area:
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            return _trim_mask(mask)
    return np.ones((1, 1), dtype=bool)


def _mask_rows(mask: np.ndarray) -> list[int]:
    return [int(r) for r in range(mask.shape[0]) if bool(mask[r].any())]


def _mask_cols(mask: np.ndarray) -> list[int]:
    return [int(c) for c in range(mask.shape[1]) if bool(mask[:, c].any())]


def _pattern_from_pair(
    mask_a: np.ndarray, mask_b: np.ndarray, color: int, axis: str
) -> np.ndarray:
    if random.random() < 0.5:
        mask_a, mask_b = mask_b, mask_a

    gap = random.randint(2, 7)
    coords: list[tuple[int, int]] = []
    if axis == "h":
        row_a = random.choice(_mask_rows(mask_a))
        row_b = random.choice(_mask_rows(mask_b))
        rb = row_a - row_b
        cb = int(mask_a.shape[1]) + gap
    else:
        col_a = random.choice(_mask_cols(mask_a))
        col_b = random.choice(_mask_cols(mask_b))
        rb = int(mask_a.shape[0]) + gap
        cb = col_a - col_b

    for r, c in np.argwhere(mask_a):
        coords.append((int(r), int(c)))
    for r, c in np.argwhere(mask_b):
        coords.append((int(r) + rb, int(c) + cb))

    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    norm = [(r - min_r, c - min_c) for r, c in coords]
    height = max(r for r, _c in norm) + 1
    width = max(c for _r, c in norm) + 1
    pattern = np.full((height, width), -1, dtype=int)
    for r, c in norm:
        pattern[r, c] = color
    return pattern


def _nonempty_coords(pattern: np.ndarray) -> np.ndarray:
    return np.argwhere(pattern >= 0)


def _can_place(grid: np.ndarray, pattern: np.ndarray, row: int, col: int, bg: int) -> bool:
    h, w = grid.shape
    ph, pw = pattern.shape
    if row < 0 or col < 0 or row + ph > h or col + pw > w:
        return False
    coords = _nonempty_coords(pattern)
    for rr, cc in coords:
        gr, gc = row + int(rr), col + int(cc)
        if int(grid[gr, gc]) != bg:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
                    return False
    return True


def _place_pattern(grid: np.ndarray, pattern: np.ndarray, bg: int) -> bool:
    h, w = grid.shape
    ph, pw = pattern.shape
    if ph > h or pw > w:
        return False
    for _ in range(PLACE_RETRIES):
        row = random.randint(0, h - ph)
        col = random.randint(0, w - pw)
        if _can_place(grid, pattern, row, col, bg):
            mask = pattern >= 0
            window = grid[row : row + ph, col : col + pw]
            window[mask] = pattern[mask]
            return True
    return False


def _component_count_for_color(grid: np.ndarray, color: int) -> int:
    bg = -1
    temp = np.full(grid.shape, bg, dtype=int)
    temp[grid == color] = color
    return len(_components_by_color(temp, bg).get(color, []))


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(N_COLORS)
    colors = [c for c in range(N_COLORS) if c != bg]
    pair_count = random.randint(2, 4)
    random.shuffle(colors)
    colors = colors[:pair_count]

    orientations = ["h", "v"]
    while len(orientations) < pair_count:
        orientations.append(random.choice(["h", "v"]))
    random.shuffle(orientations)

    patterns: list[np.ndarray] = []
    areas: set[int] = set()
    for color, axis in zip(colors, orientations):
        for _ in range(80):
            mask_a = _random_fragment()
            mask_b = _random_fragment()
            area = int(mask_a.sum() + mask_b.sum())
            if area in areas:
                continue
            pattern = _pattern_from_pair(mask_a, mask_b, color, axis)
            if max(pattern.shape) <= 13:
                areas.add(area)
                patterns.append(pattern)
                break
        else:
            return None

    height = random.randint(14, 30)
    width = random.randint(14, 30)
    grid = np.full((height, width), bg, dtype=int)
    random.shuffle(patterns)
    for pattern in patterns:
        if not _place_pattern(grid, pattern, bg):
            return None

    for color in colors:
        if _component_count_for_color(grid, color) != 2:
            return None

    if _background_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
