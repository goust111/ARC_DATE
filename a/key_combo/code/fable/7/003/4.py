from __future__ import annotations

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = spec_from_file_location("_arc_utils_fable_7_003_4", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
paste = _ARC_UTILS.paste
random_sprite = _ARC_UTILS.random_sprite


GEN_ATTEMPTS = 400
DEFECT_ATTEMPTS = 250
PALETTE = tuple(int(color) for color in Color.NOT_BLACK)


def _orbit_positions(r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (int(r), int(c)),
                (int(h - 1 - r), int(c)),
                (int(r), int(w - 1 - c)),
                (int(h - 1 - r), int(w - 1 - c)),
            }
        )
    )


def _render_expected(clean_mosaic: np.ndarray, defect_coords: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(clean_mosaic.shape, Color.BLACK, dtype=int)
    for r, c in defect_coords:
        out[int(r), int(c)] = int(clean_mosaic[int(r), int(c)])
    out_box = bbox(out != Color.BLACK)
    assert out_box is not None
    return crop_bbox(out, out_box)


def _extract_defects(mosaic: np.ndarray) -> list[tuple[tuple[int, int], int]]:
    h, w = mosaic.shape
    seen: set[tuple[int, int]] = set()
    defects: list[tuple[tuple[int, int], int]] = []

    for r in range(h):
        for c in range(w):
            coord = (int(r), int(c))
            if coord in seen:
                continue

            orbit = _orbit_positions(r, c, h, w)
            seen.update(orbit)
            colors = [int(mosaic[rr, cc]) for rr, cc in orbit]
            counts = Counter(colors)

            if len(orbit) < 4:
                assert len(counts) == 1, "axis or center orbits must stay symmetric"
                continue

            if len(counts) == 1:
                continue

            assert len(counts) == 2 and sorted(counts.values()) == [1, 3], "defect orbit must be 3:1"
            majority_color = next(color for color, count in counts.items() if count == 3)
            minority = [pos for pos in orbit if counts[int(mosaic[pos])] == 1]
            assert len(minority) == 1
            defects.append((minority[0], int(majority_color)))

    return defects


def _seed_pattern(seed_h: int, seed_w: int, palette: tuple[int, ...]) -> np.ndarray:
    seed = np.full((seed_h, seed_w), random.choice(palette), dtype=int)

    for _ in range(random.randint(3, 7)):
        color = random.choice(palette)
        mode = random.choice(("patch", "rect", "line", "cell"))

        if mode == "patch":
            patch_h = random.randint(1, seed_h)
            patch_w = random.randint(1, seed_w)
            patch = random_sprite(
                patch_h,
                patch_w,
                density=random.uniform(0.35, 0.75),
                color_palette=[int(color)],
                background=Color.BLACK,
            )
            top = random.randint(0, seed_h - patch_h)
            left = random.randint(0, seed_w - patch_w)
            paste(seed, patch, top, left, transparent=Color.BLACK)
        elif mode == "rect":
            r0 = random.randint(0, seed_h - 1)
            r1 = random.randint(r0, seed_h - 1)
            c0 = random.randint(0, seed_w - 1)
            c1 = random.randint(c0, seed_w - 1)
            seed[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        elif mode == "line":
            if random.random() < 0.5:
                seed[random.randint(0, seed_h - 1), :] = int(color)
            else:
                seed[:, random.randint(0, seed_w - 1)] = int(color)
        else:
            seed[random.randint(0, seed_h - 1), random.randint(0, seed_w - 1)] = int(color)

    if len(set(map(int, seed.ravel()))) < 3:
        seed[0, 0] = int(palette[0])
        seed[-1, -1] = int(palette[-1])
        seed[seed_h // 2, seed_w // 2] = int(palette[len(palette) // 2])

    return seed


def _mirror_seed(seed: np.ndarray, h: int, w: int) -> np.ndarray:
    row_index = np.minimum(np.arange(h), np.arange(h)[::-1])
    col_index = np.minimum(np.arange(w), np.arange(w)[::-1])
    return seed[row_index[:, None], col_index[None, :]].astype(int)


def _build_clean_mosaic(h: int, w: int) -> np.ndarray:
    palette = tuple(random.sample(PALETTE, k=random.randint(4, 7)))
    seed_h = (h + 1) // 2
    seed_w = (w + 1) // 2
    seed = _seed_pattern(seed_h, seed_w, palette)
    mosaic = _mirror_seed(seed, h, w)
    assert np.all(mosaic != Color.BLACK)
    assert len(set(map(int, mosaic.ravel()))) >= 3
    return mosaic


def _sample_defect_mask(max_h: int, max_w: int) -> np.ndarray:
    style = random.random()
    if style < 0.12:
        return np.ones((1, random.randint(2, max_w)), dtype=bool)
    if style < 0.24:
        return np.ones((random.randint(2, max_h), 1), dtype=bool)
    if style < 0.32:
        return np.ones((1, 1), dtype=bool)

    mask_h = random.randint(2, max_h)
    mask_w = random.randint(2, max_w)
    for _ in range(12):
        try:
            patch = random_sprite(
                mask_h,
                mask_w,
                density=random.uniform(0.35, 0.7),
                color_palette=[1],
                background=Color.BLACK,
            )
            return patch != Color.BLACK
        except AssertionError:
            continue

    fallback = np.ones((mask_h, mask_w), dtype=bool)
    if mask_h >= 3 and mask_w >= 3:
        fallback[1:-1, 1:-1] = False
        fallback[mask_h // 2, mask_w // 2] = True
    return fallback


def _choose_defect_coords(mosaic: np.ndarray) -> list[tuple[int, int]]:
    h, w = mosaic.shape
    max_mask_h = min(8, h)
    max_mask_w = min(8, w)

    for _ in range(DEFECT_ATTEMPTS):
        mask = _sample_defect_mask(max_mask_h, max_mask_w)
        mask_h, mask_w = mask.shape
        top = random.randint(0, h - mask_h)
        left = random.randint(0, w - mask_w)
        coords = [
            (int(top + r), int(left + c))
            for r, c in np.argwhere(mask)
        ]
        if len(coords) == 0:
            continue

        keys = set()
        valid = True
        for r, c in coords:
            orbit = _orbit_positions(r, c, h, w)
            if len(orbit) != 4:
                valid = False
                break
            key = tuple(orbit)
            if key in keys:
                valid = False
                break
            keys.add(key)
        if not valid:
            continue

        expected_colors = {int(mosaic[r, c]) for r, c in coords}
        if len(coords) >= 4 and len(expected_colors) < 2:
            continue
        return coords

    raise RuntimeError("failed to place a non-degenerate defect sprite")


def _corrupt_mosaic(clean_mosaic: np.ndarray, defect_coords: list[tuple[int, int]]) -> np.ndarray:
    corrupted = clean_mosaic.copy()
    for r, c in defect_coords:
        majority = int(clean_mosaic[int(r), int(c)])
        wrong_choices = [color for color in PALETTE if int(color) != majority]
        corrupted[int(r), int(c)] = int(random.choice(wrong_choices))
    return corrupted


def _generate_case(return_meta: bool = False):
    for _ in range(GEN_ATTEMPTS):
        canvas_h = random.randint(12, 30)
        canvas_w = random.randint(12, 30)
        mosaic_h = random.randint(7, canvas_h - 2)
        mosaic_w = random.randint(7, canvas_w - 2)
        top = random.randint(1, canvas_h - mosaic_h - 1)
        left = random.randint(1, canvas_w - mosaic_w - 1)

        if top * 2 + mosaic_h == canvas_h and left * 2 + mosaic_w == canvas_w:
            continue

        clean_mosaic = _build_clean_mosaic(mosaic_h, mosaic_w)
        defect_coords = _choose_defect_coords(clean_mosaic)
        corrupted_mosaic = _corrupt_mosaic(clean_mosaic, defect_coords)
        expected = _render_expected(clean_mosaic, defect_coords)

        grid = np.full((canvas_h, canvas_w), Color.BLACK, dtype=int)
        paste(grid, corrupted_mosaic, top, left)

        meta = {
            "clean_mosaic": clean_mosaic,
            "defect_coords": tuple(defect_coords),
            "expected": expected,
            "mosaic_bbox": (top, left, top + mosaic_h - 1, left + mosaic_w - 1),
        }
        return (grid, meta) if return_meta else grid

    raise RuntimeError("failed to generate a valid ARC instance")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)


def main(input_grid: np.ndarray) -> np.ndarray:
    """Use biaxial symmetry to recover the intended colors of the minority defect pixels and crop them as the output sprite.

    Input structure:
    - A black background surrounds one full non-black rectangular mosaic whose hidden clean form is symmetric under both horizontal and vertical reflection.
    - The mosaic colors are mirrored about the center of its non-black bounding box, not necessarily about the whole canvas.
    - Some 4-cell symmetry orbits contain exactly one recolored non-black defect pixel, making that pixel the orbit minority.

    Rule:
    - Find the tight non-black bounding box and use its center lines as the two reflection axes.
    - Partition mosaic cells into their reflection orbits, and in each 4-cell orbit identify the unique minority pixel against the orbit majority color.
    - Paint only those minority positions on a black canvas using the orbit majority color, then return the tight crop of that painted defect set.

    Key insight:
    - The answer is not the repaired mosaic: symmetry is only a detector that reveals where the wrong pixels sit and what their true colors should have been.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    mosaic_box = bbox(grid != Color.BLACK)
    assert mosaic_box is not None, "expected one non-black mosaic"
    mosaic = crop_bbox(grid, mosaic_box)
    assert np.all(mosaic != Color.BLACK), "the non-black bbox should be the full mosaic rectangle"

    defects = _extract_defects(mosaic)
    assert defects, "expected at least one defect orbit"

    out = np.full(mosaic.shape, Color.BLACK, dtype=int)
    for (r, c), majority_color in defects:
        out[int(r), int(c)] = int(majority_color)

    out_box = bbox(out != Color.BLACK)
    assert out_box is not None
    return crop_bbox(out, out_box)
