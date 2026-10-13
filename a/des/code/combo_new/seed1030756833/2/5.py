from __future__ import annotations

from collections import deque
import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 400


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _same_color_components(grid: np.ndarray, bg: int, blocked: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = blocked.copy()
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _find_scale(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = map(int, grid.shape)
    for r in range(h - 1, -1, -1):
        cols = np.flatnonzero(grid[r] != bg)
        if (
            len(cols) == 5
            and int(cols[-1] - cols[0]) == 4
            and len({int(grid[r, c]) for c in cols}) == 5
        ):
            mask = np.zeros((h, w), dtype=bool)
            mask[r, cols] = True
            return [int(grid[r, c]) for c in cols], mask

    for r in range(h - 1, -1, -1):
        c = 0
        while c < w:
            if int(grid[r, c]) == bg:
                c += 1
                continue
            start = c
            while c < w and int(grid[r, c]) != bg:
                c += 1
            end = c
            if end - start == 5:
                colors = [int(grid[r, cc]) for cc in range(start, end)]
                if len(set(colors)) == 5:
                    mask = np.zeros((h, w), dtype=bool)
                    mask[r, start:end] = True
                    return colors, mask
    raise ValueError("expected one bottom horizontal 5-color scale")


def _internal_hole_count(grid: np.ndarray, coords: np.ndarray, bg: int) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    object_mask = np.zeros_like(crop, dtype=bool)
    for r, c in coords:
        object_mask[int(r) - r0, int(c) - c0] = True

    holes = 0
    for pts in _mask_components(crop == bg):
        if any(
            int(r) == 0
            or int(c) == 0
            or int(r) == crop.shape[0] - 1
            or int(c) == crop.shape[1] - 1
            for r, c in pts
        ):
            continue
        enclosed_by_this_object = True
        for r, c in pts:
            rr, cc = int(r), int(c)
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < crop.shape[0] and 0 <= nc < crop.shape[1]:
                    if int(crop[nr, nc]) != bg and not object_mask[nr, nc]:
                        enclosed_by_this_object = False
                        break
            if not enclosed_by_this_object:
                break
        if enclosed_by_this_object:
            holes += 1
    return int(holes)


def _paint_stack(
    out: np.ndarray,
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    copies: int,
    color: int,
) -> None:
    h, w = map(int, out.shape)
    r0, c0, r1, _c1 = bbox
    shape_h = int(r1 - r0 + 1)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    for k in range(int(copies)):
        top = int(r0 - k * shape_h)
        for rr, cc in rel:
            tr, tc = top + rr, int(c0 + cc)
            if 0 <= tr < h and 0 <= tc < w:
                out[tr, tc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack each foreground sprite upward by its number of enclosed holes and recolor it from a bottom scale.

    Input structure:
    - The dominant color is the background.
    - Several separated monochrome foreground objects appear above the bottom of the grid; each may contain 0 to 4 enclosed background holes.
    - A bottom horizontal legend is a contiguous run of five distinct single pixels, read left to right as the colors for hole counts 0, 1, 2, 3, and 4.

    Rule:
    - Excluding the legend, find each same-color connected foreground object and count its enclosed background components.
    - If an object has H holes, draw H+1 copies of its exact mask in the output: the lowest copy stays at the original position and each next copy is placed directly one object-bbox height above it.
    - Every copy of that object is recolored to the H-th legend color; the legend itself is removed to background.

    Key insight:
    The hidden object property is the hole count, and the same count independently selects both the stack height and the legend color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    scale, scale_mask = _find_scale(g, bg)

    out = np.full_like(g, int(bg))
    for comp in _same_color_components(g, bg, scale_mask):
        holes = _internal_hole_count(g, comp["coords"], bg)
        if not 0 <= holes < len(scale):
            raise ValueError("object hole count must be between 0 and 4")
        _paint_stack(out, comp["coords"], comp["bbox"], holes + 1, scale[holes])
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _mask_components(mask)
    return len(comps) == 1 and int(comps[0].shape[0]) == int(mask.sum())


def _hole_count_in_mask(mask: np.ndarray) -> int:
    mask = _trim_mask(mask)
    holes = 0
    for pts in _mask_components(~mask):
        if any(
            int(r) == 0
            or int(c) == 0
            or int(r) == mask.shape[0] - 1
            or int(c) == mask.shape[1] - 1
            for r, c in pts
        ):
            continue
        holes += 1
    return int(holes)


def _zero_hole_shape() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    mask = np.ones((h, w), dtype=bool)
    for _ in range(random.randint(0, 3)):
        boundary = [
            (r, c)
            for r in range(mask.shape[0])
            for c in range(mask.shape[1])
            if mask[r, c] and (r in (0, mask.shape[0] - 1) or c in (0, mask.shape[1] - 1))
        ]
        random.shuffle(boundary)
        changed = False
        for r, c in boundary:
            if int(mask.sum()) <= 3:
                break
            trial = mask.copy()
            trial[r, c] = False
            trial = _trim_mask(trial)
            if _mask_connected(trial) and _hole_count_in_mask(trial) == 0:
                mask = trial
                changed = True
                break
        if not changed:
            break
    return _trim_mask(mask)


def _template_shape(holes: int) -> np.ndarray:
    if holes == 1:
        if random.random() < 0.5:
            return np.array(
                [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ],
                dtype=bool,
            )
        return np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        )
    if holes == 2:
        return np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
    if holes == 3:
        return np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
    if holes == 4:
        return np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
    return _zero_hole_shape()


def _shape_for_holes(holes: int) -> np.ndarray:
    for _ in range(50):
        mask = _template_shape(int(holes))
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randint(0, 3)).copy()
        mask = _trim_mask(mask)
        if _mask_connected(mask) and _hole_count_in_mask(mask) == int(holes):
            return mask
    raise RuntimeError("failed to build requested hole shape")


def _assign_object_colors(bg: int, palette: list[int], holes: list[int]) -> list[int]:
    remaining = [c for c in range(10) if c != int(bg)]
    random.shuffle(remaining)
    colors: list[int] = []
    for h in holes:
        choices = [c for c in remaining if c != int(palette[int(h)])]
        if not choices:
            choices = list(remaining)
        color = random.choice(choices)
        remaining.remove(color)
        colors.append(int(color))
    return colors


def _distribute_slack(slack: int, slots: int) -> list[int]:
    values = [0 for _ in range(slots)]
    for _ in range(int(slack)):
        values[random.randrange(slots)] += 1
    return values


def _expected_from_plan(
    shape: tuple[int, int],
    bg: int,
    palette: list[int],
    objects: list[dict],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for obj in objects:
        mask = obj["mask"]
        top, left, holes = int(obj["top"]), int(obj["left"]), int(obj["holes"])
        color = int(palette[holes])
        rel = np.argwhere(mask)
        for k in range(holes + 1):
            rr0 = top - k * int(mask.shape[0])
            for rr, cc in rel:
                out[rr0 + int(rr), left + int(cc)] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    palette = random.sample([c for c in range(10) if c != bg], 5)

    holes = list(range(5))
    if random.random() < 0.35:
        holes.append(random.choice((0, 1, 2)))
    random.shuffle(holes)

    masks = [_shape_for_holes(h) for h in holes]
    min_width = sum(int(m.shape[1]) for m in masks) + len(masks) - 1
    if min_width > 30:
        holes = [h for h, _m in zip(holes, masks)][:5]
        masks = masks[:5]
        min_width = sum(int(m.shape[1]) for m in masks) + len(masks) - 1
        if min_width > 30:
            return None

    grid_h = random.randint(27, 30)
    grid_w = random.randint(max(24, min_width), 30)
    internal_gap_base = len(masks) - 1
    slack = grid_w - (sum(int(m.shape[1]) for m in masks) + internal_gap_base)
    gaps = _distribute_slack(slack, len(masks) + 1)
    for i in range(1, len(gaps) - 1):
        gaps[i] += 1

    object_colors = _assign_object_colors(bg, palette, holes)
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    objects: list[dict] = []
    left = gaps[0]
    for i, (hole_count, mask) in enumerate(zip(holes, masks)):
        sh, sw = map(int, mask.shape)
        min_top = int(hole_count) * sh
        max_top = grid_h - sh - 2
        if min_top > max_top:
            return None
        if random.random() < 0.65:
            top = random.randint(max(min_top, max_top - 3), max_top)
        else:
            top = random.randint(min_top, max_top)
        rr, cc = np.where(mask)
        grid[top + rr, left + cc] = int(object_colors[i])
        objects.append({"holes": int(hole_count), "mask": mask, "top": int(top), "left": int(left)})
        left += sw + gaps[i + 1]

    scale_start = random.randint(0, grid_w - 5)
    grid[grid_h - 1, scale_start : scale_start + 5] = np.array(palette, dtype=int)

    expected = _expected_from_plan(grid.shape, bg, palette, objects)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
