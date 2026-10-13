from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists():
    utils_path = str(_UTILS_DIR)
    if utils_path not in sys.path:
        sys.path.append(utils_path)

try:
    from arc_utils import Color, crop_bbox, find_components, get_bg, paste
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        mask: np.ndarray | None = None,
        overwrite: bool = True,
        clip: bool = False,
    ) -> np.ndarray:
        h, w = map(int, patch.shape)
        r0 = int(r0)
        c0 = int(c0)
        if clip:
            raise ValueError("fallback paste does not support clip=True")
        if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
            raise ValueError("paste out of bounds")
        view = dst[r0 : r0 + h, c0 : c0 + w]
        write_mask = np.ones((h, w), dtype=bool)
        if mask is not None:
            write_mask &= np.asarray(mask, dtype=bool)
        if transparent is not None:
            write_mask &= patch != int(transparent)
            if not overwrite:
                write_mask &= view == int(transparent)
        elif not overwrite:
            raise ValueError("overwrite=False requires transparent")
        view[write_mask] = patch[write_mask]
        return dst

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)
        if connectivity != 4:
            raise ValueError("fallback find_components only supports connectivity=4")

        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        out: list[dict] = []

        allowed = None if colors is None else {int(c) for c in colors}

        for r in range(h):
            for c in range(w):
                if visited[r, c]:
                    continue
                color = int(grid[r, c])
                if color == bg:
                    continue
                if allowed is not None and color not in allowed:
                    continue

                visited[r, c] = True
                q = deque([(r, c)])
                coords: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for dr, dc in dirs:
                        nr = rr + dr
                        nc = cc + dc
                        if not (0 <= nr < h and 0 <= nc < w) or visited[nr, nc]:
                            continue
                        ncolor = int(grid[nr, nc])
                        if ncolor == bg:
                            continue
                        if allowed is not None and ncolor not in allowed:
                            continue
                        if monochromatic and ncolor != color:
                            continue
                        visited[nr, nc] = True
                        q.append((nr, nc))

                if len(coords) < int(min_area):
                    continue
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": color if monochromatic else None,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(len(coords)),
                    }
                )
        return out


BG = int(Color.BLACK)
COLOR_POOL = [int(c) for c in Color.NOT_BLACK]
ORIENTATIONS = ("identity", "flip_lr", "flip_ud", "rot180")
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500
SPRITE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 200
CLUE_ORDER_ATTEMPTS = 40
INDEPENDENT_SET_ATTEMPTS = 32
MIN_RESIDUES = 3
MAX_RESIDUES = 6
MIN_GAP = 2
_GEN_CALL_COUNT = 0


def _transform_patch(patch: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "identity":
        return np.asarray(patch, dtype=int).copy()
    if orientation == "flip_lr":
        return np.fliplr(patch).copy()
    if orientation == "flip_ud":
        return np.flipud(patch).copy()
    if orientation == "rot180":
        return np.rot90(patch, 2).copy()
    raise ValueError(f"unknown orientation: {orientation}")


def _patch_cells(patch: np.ndarray) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(np.asarray(patch) != BG):
        cells.append((int(r), int(c), int(patch[r, c])))
    return cells


def _candidate_signature(
    top_left: tuple[int, int], cells: list[tuple[int, int, int]]
) -> tuple[tuple[int, int, int], ...]:
    r0, c0 = map(int, top_left)
    return tuple(sorted((r0 + r, c0 + c, color) for r, c, color in cells))


def _enumerate_candidates(
    shape: tuple[int, int],
    sprite_patch: np.ndarray,
    residues: list[tuple[int, int, int]],
) -> list[dict]:
    assert residues, "expected at least one residue pixel"
    h, w = map(int, shape)
    results: list[dict] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()

    for orientation in ORIENTATIONS:
        oriented = _transform_patch(sprite_patch, orientation)
        ph, pw = map(int, oriented.shape)
        cells = _patch_cells(oriented)
        local_map = {(r, c): color for r, c, color in cells}
        by_color: dict[int, list[tuple[int, int]]] = {}
        for r, c, color in cells:
            by_color.setdefault(int(color), []).append((int(r), int(c)))

        anchor = min(residues, key=lambda item: len(by_color.get(int(item[2]), [])))
        anchor_matches = by_color.get(int(anchor[2]), [])
        if not anchor_matches:
            continue

        for ar, ac in anchor_matches:
            r0 = int(anchor[0]) - int(ar)
            c0 = int(anchor[1]) - int(ac)
            if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
                continue

            ok = True
            for rr, cc, color in residues:
                lr = int(rr) - r0
                lc = int(cc) - c0
                if local_map.get((lr, lc)) != int(color):
                    ok = False
                    break
            if not ok:
                continue

            sig = _candidate_signature((r0, c0), cells)
            if sig in seen:
                continue
            seen.add(sig)
            results.append(
                {
                    "orientation": orientation,
                    "top_left": (r0, c0),
                    "patch": oriented,
                    "cells": cells,
                    "signature": sig,
                }
            )

    return results


def _extract_instance(
    grid: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected foreground cells"
    sprite_comps = [comp for comp in comps if int(comp["area"]) > 1]
    assert len(sprite_comps) == 1, "expected exactly one connected multicolor sprite"
    sprite_comp = sprite_comps[0]
    sprite_patch = crop_bbox(g, sprite_comp["bbox"])

    residues: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is sprite_comp:
            continue
        assert int(comp["area"]) == 1, "residue pixels must be isolated singletons"
        r, c = map(int, comp["coords"][0])
        residues.append((r, c, int(g[r, c])))

    assert residues, "expected at least one residue pixel"
    return bg, g, sprite_patch, residues


def _rect_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    return not (
        a[2] + int(pad) < b[0]
        or b[2] + int(pad) < a[0]
        or a[3] + int(pad) < b[1]
        or b[3] + int(pad) < a[1]
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_connected_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    r0 = random.randrange(height)
    c0 = random.randrange(width)
    mask[r0, c0] = True

    while int(mask.sum()) < int(target_area):
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIRS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True

    return _crop_mask(mask)


def _build_sprite_patch() -> np.ndarray | None:
    for _ in range(SPRITE_ATTEMPTS):
        height = random.randint(4, 7)
        width = random.randint(4, 7)
        max_area = min(height * width - 1, 14)
        min_area = max(7, (height * width) // 2)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _random_connected_mask(height, width, area)
        mh, mw = map(int, mask.shape)
        if mh < 3 or mw < 3 or int(mask.sum()) < 7:
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        color_count = random.randint(3, min(5, len(coords)))
        palette = random.sample(COLOR_POOL, color_count)
        colors = palette[:]
        while len(colors) < len(coords):
            colors.append(random.choice(palette))
        random.shuffle(colors)

        patch = np.zeros(mask.shape, dtype=int)
        for (r, c), color in zip(coords, colors):
            patch[int(r), int(c)] = int(color)

        signatures = {
            tuple(map(int, _transform_patch(patch, orientation).ravel()))
            for orientation in ORIENTATIONS
        }
        if len(signatures) != len(ORIENTATIONS):
            continue
        return patch
    return None


def _best_sparse_pool(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    best: list[tuple[int, int, int]] = []
    for _ in range(INDEPENDENT_SET_ATTEMPTS):
        order = cells[:]
        random.shuffle(order)
        chosen: list[tuple[int, int, int]] = []
        for cell in order:
            r, c, _ = cell
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc, _ in chosen):
                chosen.append(cell)
        if len(chosen) > len(best):
            best = chosen
    return best


def _place_two_copies(
    canvas_shape: tuple[int, int], sprite_shape: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, canvas_shape)
    ph, pw = map(int, sprite_shape)
    for _ in range(PLACEMENT_ATTEMPTS):
        sprite_r = random.randint(0, h - ph)
        sprite_c = random.randint(0, w - pw)
        copy_r = random.randint(0, h - ph)
        copy_c = random.randint(0, w - pw)
        sprite_box = (sprite_r, sprite_c, sprite_r + ph - 1, sprite_c + pw - 1)
        copy_box = (copy_r, copy_c, copy_r + ph - 1, copy_c + pw - 1)
        if _rect_overlap(sprite_box, copy_box, pad=MIN_GAP):
            continue
        return (sprite_r, sprite_c), (copy_r, copy_c)
    return None


def _render_with_residues(
    shape: tuple[int, int],
    sprite_patch: np.ndarray,
    sprite_top_left: tuple[int, int],
    residue_top_left: tuple[int, int],
    residues_local: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    paste(grid, sprite_patch, int(sprite_top_left[0]), int(sprite_top_left[1]), transparent=BG)
    for r, c, color in residues_local:
        grid[int(residue_top_left[0]) + int(r), int(residue_top_left[1]) + int(c)] = int(color)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the unique rigid placement of a hidden sprite copy from sparse true-color residues and draw the full copy.

    Input structure:
    - A black background contains one connected multicolor sprite shown in full.
    - Elsewhere, several isolated colored singleton pixels are sparse residues of another copy of that sprite.
    - The hidden copy uses exactly one of four poses: original, left-right mirror, up-down mirror, or 180-degree rotation.

    Rule:
    - Crop the visible sprite as the source pattern.
    - Among the four allowed poses and all in-bounds translations, find the unique placement whose sprite pixels cover every residue with the exact same position-relative color.
    - Draw that entire transformed copy onto the grid, while keeping the original sprite and the residue pixels intact.

    Key insight:
    - The scattered pixels are not standalone markers; they are a sparse sample of one unseen transformed sprite, so only one pose-plus-offset explains all residue colors at once.
    """
    bg, g, sprite_patch, residues = _extract_instance(grid)
    assert bg == BG, "expected a black background"

    candidates = _enumerate_candidates(g.shape, sprite_patch, residues)
    assert len(candidates) == 1, "residues must pin exactly one transformed placement"
    candidate = candidates[0]

    out = g.copy()
    for rr, cc, color in candidate["signature"]:
        out[int(rr), int(cc)] = int(color)
    return out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    target_orientation = ORIENTATIONS[_GEN_CALL_COUNT % len(ORIENTATIONS)]

    for _ in range(GEN_ATTEMPTS):
        sprite_patch = _build_sprite_patch()
        if sprite_patch is None:
            continue

        transformed = _transform_patch(sprite_patch, target_orientation)
        transformed_cells = _patch_cells(transformed)
        sparse_pool = _best_sparse_pool(transformed_cells)
        if len(sparse_pool) < MIN_RESIDUES:
            continue

        ph, pw = map(int, sprite_patch.shape)
        min_h = max(12, ph * 2 + MIN_GAP + 1)
        min_w = max(12, pw * 2 + MIN_GAP + 1)
        if min_h > 30 or min_w > 30:
            continue
        grid_h = random.randint(min_h, min(30, min_h + 8))
        grid_w = random.randint(min_w, min(30, min_w + 8))

        placements = _place_two_copies((grid_h, grid_w), sprite_patch.shape)
        if placements is None:
            continue
        sprite_top_left, residue_top_left = placements

        true_signature = _candidate_signature(residue_top_left, transformed_cells)
        accepted_grid = None

        for _ in range(CLUE_ORDER_ATTEMPTS):
            order = sparse_pool[:]
            random.shuffle(order)
            chosen: list[tuple[int, int, int]] = []
            prev_count = None

            for cell in order[:MAX_RESIDUES]:
                chosen.append(cell)
                grid = _render_with_residues(
                    (grid_h, grid_w),
                    sprite_patch,
                    sprite_top_left,
                    residue_top_left,
                    chosen,
                )
                candidates = _enumerate_candidates(grid.shape, sprite_patch, [
                    (residue_top_left[0] + r, residue_top_left[1] + c, color)
                    for r, c, color in chosen
                ])
                count = len(candidates)
                assert count >= 1, "the true copy must remain valid"

                if count == 1:
                    if (
                        len(chosen) >= MIN_RESIDUES
                        and prev_count is not None
                        and prev_count > 1
                        and len(chosen) < len(transformed_cells)
                        and candidates[0]["signature"] == true_signature
                        and len({color for _r, _c, color in chosen}) >= 2
                    ):
                        accepted_grid = grid
                    break

                prev_count = count

            if accepted_grid is not None:
                break

        if accepted_grid is None:
            continue

        solved = main(accepted_grid)
        expected = accepted_grid.copy()
        for rr, cc, color in true_signature:
            expected[int(rr), int(cc)] = int(color)

        if not np.array_equal(solved, expected):
            continue

        _GEN_CALL_COUNT += 1
        return accepted_grid

    raise RuntimeError("failed to generate a valid residue-placement puzzle")
