from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class RingModel:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    thickness: int
    side: str
    gap_start: int
    gap_width: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width


SIDES = ("top", "bottom", "left", "right")


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_band(shape: tuple[int, int], thickness: int) -> np.ndarray:
    h, w = shape
    assert 1 <= thickness <= min((h - 1) // 2, (w - 1) // 2)
    band = np.zeros((h, w), dtype=bool)
    band[:thickness, :] = True
    band[h - thickness :, :] = True
    band[:, :thickness] = True
    band[:, w - thickness :] = True
    return band


def _gap_bounds_local(
    shape: tuple[int, int], thickness: int, side: str, start: int, width: int
) -> tuple[int, int, int, int]:
    h, w = shape
    if side == "top":
        return 0, thickness, start, start + width
    if side == "bottom":
        return h - thickness, h, start, start + width
    if side == "left":
        return start, start + width, 0, thickness
    if side == "right":
        return start, start + width, w - thickness, w
    raise ValueError(f"unknown side: {side}")


def _gap_mask(
    shape: tuple[int, int], thickness: int, side: str, start: int, width: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, r1, c0, c1 = _gap_bounds_local(shape, thickness, side, start, width)
    mask[r0:r1, c0:c1] = True
    return mask


def _fit_ring_model(color: int, mask: np.ndarray) -> RingModel:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    max_thickness = min((h - 1) // 2, (w - 1) // 2)
    candidates: list[RingModel] = []

    for thickness in range(1, max_thickness + 1):
        band = _rect_band((h, w), thickness)
        if np.any(sub & ~band):
            continue
        missing = band & ~sub
        if not np.any(missing):
            continue
        if np.sum(missing) > thickness * 3:
            continue

        for side in SIDES:
            side_len = w if side in ("top", "bottom") else h
            max_gap = min(3, side_len)
            for gap_width in range(1, max_gap + 1):
                for gap_start in range(side_len - gap_width + 1):
                    if np.array_equal(
                        missing,
                        _gap_mask((h, w), thickness, side, gap_start, gap_width),
                    ):
                        candidates.append(
                            RingModel(
                                color=color,
                                r0=r0,
                                c0=c0,
                                r1=r1,
                                c1=c1,
                                thickness=thickness,
                                side=side,
                                gap_start=gap_start,
                                gap_width=gap_width,
                            )
                        )

    assert len(candidates) == 1, f"ambiguous ring model for color {color}"
    return candidates[0]


def _sort_nested(models: list[RingModel]) -> list[RingModel]:
    ordered = sorted(models, key=lambda model: model.area)
    for inner, outer in zip(ordered, ordered[1:]):
        assert outer.r0 < inner.r0
        assert outer.c0 < inner.c0
        assert outer.r1 > inner.r1
        assert outer.c1 > inner.c1
    return ordered


def _gap_bounds_abs(model: RingModel) -> tuple[int, int, int, int]:
    local = _gap_bounds_local(
        (model.height, model.width),
        model.thickness,
        model.side,
        model.gap_start,
        model.gap_width,
    )
    lr0, lr1, lc0, lc1 = local
    return model.r0 + lr0, model.r0 + lr1, model.c0 + lc0, model.c0 + lc1


def _paint_ring(grid: np.ndarray, model: RingModel, fill_color: int) -> None:
    r0, c0, r1, c1 = model.r0, model.c0, model.r1, model.c1
    t = model.thickness
    grid[r0 : r0 + t, c0 : c1 + 1] = model.color
    grid[r1 - t + 1 : r1 + 1, c0 : c1 + 1] = model.color
    grid[r0 : r1 + 1, c0 : c0 + t] = model.color
    grid[r0 : r1 + 1, c1 - t + 1 : c1 + 1] = model.color
    gr0, gr1, gc0, gc1 = _gap_bounds_abs(model)
    grid[gr0:gr1, gc0:gc1] = fill_color


def _expected_output(grid: np.ndarray, bg: int, models: list[RingModel]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for i, model in enumerate(models):
        fill_color = bg if i == 0 else models[i - 1].color
        gr0, gr1, gc0, gc1 = _gap_bounds_abs(model)
        out[gr0:gr1, gc0:gc1] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the nesting order of gapped rectangular rings and fill each gap with
    the adjacent inner ring's color. The innermost gap is filled with the
    background color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert 3 <= len(colors) <= 5, "expected 3 to 5 ring colors"

    models = _sort_nested([_fit_ring_model(color, g == color) for color in colors])

    out = g.copy()
    for i, model in enumerate(models):
        fill_color = bg if i == 0 else models[i - 1].color
        gr0, gr1, gc0, gc1 = _gap_bounds_abs(model)
        patch = out[gr0:gr1, gc0:gc1]
        patch[patch == bg] = fill_color
        out[gr0:gr1, gc0:gc1] = patch

    return out


# --------------- generator (nested rectangular ring bands) ---------------

COLOR_VALUES = tuple(range(10))
GRID_MAX = 30
RING_COUNT_RANGE = (3, 5)
MARGIN_RANGE = (1, 4)
INNER_SPAN_RANGE = (2, 8)
GEN_RETRIES = 400


def _sample_thicknesses(rng: random.Random, ring_count: int) -> list[int]:
    thicknesses = [rng.choice((1, 2)) for _ in range(ring_count)]
    if ring_count <= 4 and rng.random() < 0.25:
        thicknesses[rng.randrange(ring_count)] = 3
    return thicknesses


def _sample_models(rng: random.Random) -> tuple[np.ndarray, int, list[RingModel]] | None:
    ring_count = rng.randint(*RING_COUNT_RANGE)
    thicknesses = _sample_thicknesses(rng, ring_count)

    top = rng.randint(*MARGIN_RANGE)
    bottom = rng.randint(*MARGIN_RANGE)
    left = rng.randint(*MARGIN_RANGE)
    right = rng.randint(*MARGIN_RANGE)

    total_thickness = sum(thicknesses)
    max_inner_h = min(INNER_SPAN_RANGE[1], GRID_MAX - top - bottom - 2 * total_thickness)
    max_inner_w = min(INNER_SPAN_RANGE[1], GRID_MAX - left - right - 2 * total_thickness)
    if max_inner_h < INNER_SPAN_RANGE[0] or max_inner_w < INNER_SPAN_RANGE[0]:
        return None

    inner_h = rng.randint(INNER_SPAN_RANGE[0], max_inner_h)
    inner_w = rng.randint(INNER_SPAN_RANGE[0], max_inner_w)
    outer_h = inner_h + 2 * total_thickness
    outer_w = inner_w + 2 * total_thickness
    h = outer_h + top + bottom
    w = outer_w + left + right
    assert h <= GRID_MAX and w <= GRID_MAX

    bg = rng.choice(COLOR_VALUES)
    palette = [color for color in COLOR_VALUES if color != bg]
    rng.shuffle(palette)
    colors = palette[:ring_count]

    outer_r0 = top
    outer_c0 = left
    models: list[RingModel] = []
    side_history: list[str] = []

    for i in range(ring_count):
        outside_offset = sum(thicknesses[i + 1 :])
        outer_size_h = inner_h + 2 * sum(thicknesses[: i + 1])
        outer_size_w = inner_w + 2 * sum(thicknesses[: i + 1])
        r0 = outer_r0 + outside_offset
        c0 = outer_c0 + outside_offset
        r1 = r0 + outer_size_h - 1
        c1 = c0 + outer_size_w - 1
        side = rng.choice(SIDES)
        side_len = outer_size_w if side in ("top", "bottom") else outer_size_h
        max_gap = min(3, side_len - 2)
        if max_gap < 1:
            return None
        gap_width = rng.randint(1, max_gap)
        gap_start = rng.randint(1, side_len - gap_width - 1)
        side_history.append(side)
        models.append(
            RingModel(
                color=colors[i],
                r0=r0,
                c0=c0,
                r1=r1,
                c1=c1,
                thickness=thicknesses[i],
                side=side,
                gap_start=gap_start,
                gap_width=gap_width,
            )
        )

    if len(set(side_history)) == 1:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for model in models:
        _paint_ring(grid, model, bg)

    bg_count = int(np.count_nonzero(grid == bg))
    other_counts = [int(np.count_nonzero(grid == color)) for color in colors]
    if bg_count <= max(other_counts):
        return None

    fitted = _sort_nested([_fit_ring_model(model.color, grid == model.color) for model in models])
    for model, refit in zip(models, fitted):
        if model != refit:
            return None

    expected = _expected_output(grid, bg, models)
    assert np.any(expected != grid), "expected a non-trivial transformation"
    assert np.array_equal(main(grid), expected), "generator/main mismatch"
    return grid, bg, models


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        sample = _sample_models(rng)
        if sample is not None:
            grid, _, _ = sample
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
