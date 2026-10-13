from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BG = 8
CRACK = 0
OBJECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
OBJECT_COUNT_RANGE = (2, 4)
SIDE_MIN = 4
SIDE_MAX = 7
EDGE_BIAS_PROB = 0.4
MIXED_ORIENTATION_PROB = 0.6
MAX_FOREGROUND_FRACTION = 0.32
MAX_ATTEMPTS = 500
PER_OBJECT_ATTEMPTS = 120


@dataclass(frozen=True)
class RectSpec:
    r0: int
    c0: int
    height: int
    width: int
    color: int
    orientation: str
    split: int


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _dilate1(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr0 = max(0, dr)
            rr1 = mask.shape[0] + min(0, dr)
            cc0 = max(0, dc)
            cc1 = mask.shape[1] + min(0, dc)
            out[rr0:rr1, cc0:cc1] |= mask[rr0 - dr:rr1 - dr, cc0 - dc:cc1 - dc]
    return out


def _object_patch(spec: RectSpec) -> np.ndarray:
    patch = np.full((spec.height, spec.width), spec.color, dtype=int)
    if spec.orientation == "H":
        patch[spec.split, :] = CRACK
    else:
        patch[:, spec.split] = CRACK
    return patch


def _paint_object(grid: np.ndarray, spec: RectSpec) -> None:
    grid[spec.r0 : spec.r0 + spec.height, spec.c0 : spec.c0 + spec.width] = _object_patch(spec)


def _transformed_cells(spec: RectSpec, grid_h: int, grid_w: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    counts = [0, 0]
    if spec.orientation == "H":
        shift = spec.width
        for dr in range(spec.split):
            rr = spec.r0 + dr - shift
            if not (0 <= rr < grid_h):
                continue
            for dc in range(spec.width):
                cc = spec.c0 + dc
                if 0 <= cc < grid_w:
                    cells.append((rr, cc))
                    counts[0] += 1
        for dr in range(spec.split + 1, spec.height):
            rr = spec.r0 + dr + shift
            if not (0 <= rr < grid_h):
                continue
            for dc in range(spec.width):
                cc = spec.c0 + dc
                if 0 <= cc < grid_w:
                    cells.append((rr, cc))
                    counts[1] += 1
    else:
        shift = spec.height
        for dc in range(spec.split):
            cc = spec.c0 + dc - shift
            if not (0 <= cc < grid_w):
                continue
            for dr in range(spec.height):
                rr = spec.r0 + dr
                if 0 <= rr < grid_h:
                    cells.append((rr, cc))
                    counts[0] += 1
        for dc in range(spec.split + 1, spec.width):
            cc = spec.c0 + dc + shift
            if not (0 <= cc < grid_w):
                continue
            for dr in range(spec.height):
                rr = spec.r0 + dr
                if 0 <= rr < grid_h:
                    cells.append((rr, cc))
                    counts[1] += 1
    return cells, (counts[0], counts[1])


def _paint_transformed(out: np.ndarray, spec: RectSpec) -> None:
    cells, _ = _transformed_cells(spec, out.shape[0], out.shape[1])
    for r, c in cells:
        out[r, c] = spec.color


def _infer_spec(grid: np.ndarray, comp: np.ndarray) -> RectSpec:
    r0 = int(comp[:, 0].min())
    c0 = int(comp[:, 1].min())
    r1 = int(comp[:, 0].max())
    c1 = int(comp[:, 1].max())
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = sorted(set(map(int, np.unique(patch))))
    if len(colors) != 2 or CRACK not in colors or BG in colors:
        raise ValueError("component is not a cracked solid rectangle")
    color = next(col for col in colors if col != CRACK)
    if not np.all((patch == color) | (patch == CRACK)):
        raise ValueError("unexpected color inside rectangle")

    black_rows = [r for r in range(patch.shape[0]) if np.all(patch[r, :] == CRACK)]
    black_cols = [c for c in range(patch.shape[1]) if np.all(patch[:, c] == CRACK)]
    if len(black_rows) == 1 and len(black_cols) == 0:
        split = black_rows[0]
        if not (0 < split < patch.shape[0] - 1):
            raise ValueError("horizontal crack must leave two non-empty halves")
        if not np.all(patch[:split, :] == color) or not np.all(patch[split + 1 :, :] == color):
            raise ValueError("horizontal crack shape is invalid")
        return RectSpec(r0, c0, patch.shape[0], patch.shape[1], color, "H", split)
    if len(black_rows) == 0 and len(black_cols) == 1:
        split = black_cols[0]
        if not (0 < split < patch.shape[1] - 1):
            raise ValueError("vertical crack must leave two non-empty halves")
        if not np.all(patch[:, :split] == color) or not np.all(patch[:, split + 1 :] == color):
            raise ValueError("vertical crack shape is invalid")
        return RectSpec(r0, c0, patch.shape[0], patch.shape[1], color, "V", split)
    raise ValueError("expected exactly one straight crack")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full(g.shape, BG, dtype=int)
    for comp in _components4(g != BG):
        spec = _infer_spec(g, comp)
        _paint_transformed(out, spec)
    return out


def _input_mask(spec: RectSpec, grid_h: int, grid_w: int) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    mask[spec.r0 : spec.r0 + spec.height, spec.c0 : spec.c0 + spec.width] = True
    return mask


def _output_mask(spec: RectSpec, grid_h: int, grid_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    cells, counts = _transformed_cells(spec, grid_h, grid_w)
    for r, c in cells:
        mask[r, c] = True
    return mask, counts


def _sample_orientations(rng: np.random.Generator, count: int) -> list[str]:
    if count == 1:
        return ["H" if rng.random() < 0.5 else "V"]
    if rng.random() < MIXED_ORIENTATION_PROB:
        n_h = int(rng.integers(1, count))
        orientations = ["H"] * n_h + ["V"] * (count - n_h)
        rng.shuffle(orientations)
        return orientations
    return ["H"] * count if rng.random() < 0.5 else ["V"] * count


def _sample_origin(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    height: int,
    width: int,
    orientation: str,
) -> tuple[int, int]:
    max_r = grid_h - height
    max_c = grid_w - width
    r0 = int(rng.integers(0, max_r + 1))
    c0 = int(rng.integers(0, max_c + 1))
    if orientation == "H" and rng.random() < EDGE_BIAS_PROB:
        if rng.random() < 0.5:
            upper = min(max_r, width - 1)
            if upper >= 0:
                r0 = int(rng.integers(0, upper + 1))
        else:
            lower = max(0, max_r - width + 1)
            r0 = int(rng.integers(lower, max_r + 1))
    if orientation == "V" and rng.random() < EDGE_BIAS_PROB:
        if rng.random() < 0.5:
            upper = min(max_c, height - 1)
            if upper >= 0:
                c0 = int(rng.integers(0, upper + 1))
        else:
            lower = max(0, max_c - height + 1)
            c0 = int(rng.integers(lower, max_c + 1))
    return r0, c0


def _sample_spec(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    color: int,
    orientation: str,
) -> RectSpec:
    height = int(rng.integers(SIDE_MIN, SIDE_MAX + 1))
    width = int(rng.integers(SIDE_MIN, SIDE_MAX + 1))
    if orientation == "H":
        split = int(rng.integers(1, height - 1))
    else:
        split = int(rng.integers(1, width - 1))
    r0, c0 = _sample_origin(rng, grid_h, grid_w, height, width, orientation)
    return RectSpec(r0, c0, height, width, color, orientation, split)


def _render_expected(specs: list[RectSpec], grid_h: int, grid_w: int) -> np.ndarray:
    out = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        _paint_transformed(out, spec)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    grid_w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    count = int(rng.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    colors = [int(v) for v in rng.choice(OBJECT_COLORS, size=count, replace=False)]
    orientations = _sample_orientations(rng, count)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    input_forbidden = np.zeros((grid_h, grid_w), dtype=bool)
    output_forbidden = np.zeros((grid_h, grid_w), dtype=bool)
    input_occ = np.zeros((grid_h, grid_w), dtype=bool)
    output_occ = np.zeros((grid_h, grid_w), dtype=bool)
    specs: list[RectSpec] = []

    for color, orientation in zip(colors, orientations, strict=False):
        placed = False
        for _ in range(PER_OBJECT_ATTEMPTS):
            spec = _sample_spec(rng, grid_h, grid_w, color, orientation)
            in_mask = _input_mask(spec, grid_h, grid_w)
            out_mask, half_counts = _output_mask(spec, grid_h, grid_w)
            if min(half_counts) < 2:
                continue
            if not out_mask.any():
                continue
            if np.any(input_forbidden & in_mask):
                continue
            if np.any(output_forbidden & out_mask):
                continue
            specs.append(spec)
            _paint_object(grid, spec)
            input_occ |= in_mask
            output_occ |= out_mask
            input_forbidden |= _dilate1(in_mask)
            output_forbidden |= _dilate1(out_mask)
            placed = True
            break
        if not placed:
            return None

    if input_occ.mean() > MAX_FOREGROUND_FRACTION:
        return None
    if output_occ.mean() > MAX_FOREGROUND_FRACTION:
        return None

    expected = _render_expected(specs, grid_h, grid_w)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
