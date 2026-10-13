from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_shared_arc_utils", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    bbox_coords = _ARC_UTILS.bbox_coords
    crop_bbox = _ARC_UTILS.crop_bbox
else:
    class Color:
        BLACK = 0
        RED = 2
        GREEN = 3
        GREY = 5
        TEAL = 8

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


BG = int(Color.BLACK)
LINE = int(Color.GREY)
OBJ = int(Color.TEAL)
MATCH = int(Color.GREEN)
MISS = int(Color.RED)

GRID_CELLS = 4
CELL_SIZE = 3
GRID_SHAPE = (GRID_CELLS * CELL_SIZE + GRID_CELLS + 1, GRID_CELLS * CELL_SIZE + GRID_CELLS + 1)
EDGE_POSITIONS = [(r, c) for r in range(GRID_CELLS) for c in range(GRID_CELLS) if r in {0, GRID_CELLS - 1} or c in {0, GRID_CELLS - 1}]
INNER_POSITIONS = [(r, c) for r in range(1, GRID_CELLS - 1) for c in range(1, GRID_CELLS - 1)]
GEN_ATTEMPTS = 400


def _is_four_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(coords.shape[0])


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if bool(cell) else "0" for cell in row) for row in mask)


def _dihedral_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    variants: list[tuple[str, ...]] = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(_mask_key(_tight_mask(rotated)))
        variants.append(_mask_key(_tight_mask(np.fliplr(rotated))))
    return min(variants)


def _left_shear_cell(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    bottom = int(coords[:, 0].max())
    out = np.zeros_like(mask, dtype=bool)
    for r, c in coords:
        rr = int(r)
        cc = int(c) - (bottom - rr)
        if cc >= 0:
            out[rr, cc] = True
    assert out.any(), "shear should keep at least the bottom row"
    return _tight_mask(out)


def _line_positions(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == LINE)]
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == LINE)]


def _cell_spans(line_positions: list[int], limit: int) -> list[tuple[int, int]]:
    assert line_positions and line_positions[0] == 0 and line_positions[-1] == limit - 1
    spans = [(a + 1, b - 1) for a, b in zip(line_positions, line_positions[1:]) if b - a > 1]
    assert len(spans) == GRID_CELLS
    assert all((b - a + 1) == CELL_SIZE for a, b in spans)
    return spans


def _extract_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    row_spans = _cell_spans(_line_positions(g, axis=0), g.shape[0])
    col_spans = _cell_spans(_line_positions(g, axis=1), g.shape[1])

    cells: dict[tuple[int, int], np.ndarray] = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            patch = crop_bbox(g, (r0, c0, r1, c1))
            assert patch.shape == (CELL_SIZE, CELL_SIZE)
            assert np.all(np.isin(patch, [BG, OBJ]))
            mask = patch == OBJ
            area = int(mask.sum())
            assert 3 <= area <= 5
            assert _is_four_connected(mask)
            cells[(ri, ci)] = mask
    return row_spans, col_spans, cells


def _analyze_grid(grid: np.ndarray) -> dict:
    row_spans, col_spans, cells = _extract_cells(grid)
    edge_shear_sigs = set()
    edge_unsheared_sigs = set()
    edge_shear_keys = set()
    for pos, mask in cells.items():
        if pos in EDGE_POSITIONS:
            tight = _tight_mask(mask)
            sheared = _left_shear_cell(mask)
            edge_unsheared_sigs.add(_dihedral_signature(tight))
            edge_shear_sigs.add(_dihedral_signature(sheared))
            edge_shear_keys.add(_mask_key(sheared))

    inner = []
    for pos in INNER_POSITIONS:
        mask = cells[pos]
        tight = _tight_mask(mask)
        sig = _dihedral_signature(tight)
        inner.append(
            {
                "pos": pos,
                "mask": mask,
                "tight_key": _mask_key(tight),
                "sig": sig,
                "match": sig in edge_shear_sigs,
                "no_shear_match": sig in edge_unsheared_sigs,
                "exact_shear_match": _mask_key(tight) in edge_shear_keys,
            }
        )

    return {
        "row_spans": row_spans,
        "col_spans": col_spans,
        "cells": cells,
        "edge_shear_sigs": edge_shear_sigs,
        "edge_unsheared_sigs": edge_unsheared_sigs,
        "edge_shear_keys": edge_shear_keys,
        "inner": inner,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the inner cell sprites according to whether they match a sheared edge sprite up to dihedral symmetry.

    Input structure:
    - Grey lattice lines divide the grid into a 4x4 array of 3x3 cells.
    - Every cell contains one teal 4-connected sprite of area 3 to 5 on a black background.
    - Perimeter cells are edge cells; the central 2x2 block are inner cells.

    Rule:
    - For each edge-cell sprite, shear it left inside its own 3x3 cell, using the bottom bbox row as the anchor so each higher row shifts one extra step left and out-of-cell pixels are clipped.
    - Normalize each sheared edge sprite and each unsheared inner sprite to their tight masks, then compare them up to rotation and reflection.
    - Clear all edge sprites; recolor matching inner sprites green and non-matching inner sprites red; keep black backgrounds and grey lattice lines unchanged.

    Key insight:
    - The relevant comparison target is not the visible edge sprite itself but its hidden left-sheared version, and matching must ignore orientation via the dihedral symmetries.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)

    out = np.full(g.shape, BG, dtype=int)
    out[g == LINE] = LINE
    for cell in analysis["inner"]:
        ri, ci = cell["pos"]
        r0, r1 = analysis["row_spans"][ri]
        c0, c1 = analysis["col_spans"][ci]
        color = MATCH if cell["match"] else MISS
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[cell["mask"]] = color
    return out


def _enumerate_candidates() -> list[dict]:
    candidates = []
    idx = 0
    for bits in range(1 << (CELL_SIZE * CELL_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(CELL_SIZE * CELL_SIZE)], dtype=bool).reshape(CELL_SIZE, CELL_SIZE)
        area = int(mask.sum())
        if not (3 <= area <= 5):
            continue
        if not _is_four_connected(mask):
            continue
        tight = _tight_mask(mask)
        sheared = _left_shear_cell(mask)
        candidates.append(
            {
                "id": idx,
                "mask": mask,
                "raw_key": _mask_key(mask),
                "tight_key": _mask_key(tight),
                "sig": _dihedral_signature(tight),
                "sheared": sheared,
                "shear_key": _mask_key(sheared),
                "shear_sig": _dihedral_signature(sheared),
            }
        )
        idx += 1
    return candidates


CANDIDATES = _enumerate_candidates()


def _choose_one(pool: list[dict], used_raw: set[tuple[str, ...]]) -> dict | None:
    choices = [item for item in pool if item["raw_key"] not in used_raw]
    if not choices:
        return None
    item = random.choice(choices)
    used_raw.add(item["raw_key"])
    return item


def _choose_many(pool: list[dict], count: int, used_raw: set[tuple[str, ...]]) -> list[dict] | None:
    choices = pool[:]
    random.shuffle(choices)
    picked = []
    for item in choices:
        if item["raw_key"] in used_raw:
            continue
        picked.append(item)
        used_raw.add(item["raw_key"])
        if len(picked) == count:
            return picked
    return None


def _choose_inner_masks(edge_items: list[dict]) -> list[dict] | None:
    edge_raw = {item["raw_key"] for item in edge_items}
    edge_uns = {item["sig"] for item in edge_items}
    edge_shear = {item["shear_sig"] for item in edge_items}
    edge_shear_keys = {item["shear_key"] for item in edge_items}

    remaining = [item for item in CANDIDATES if item["raw_key"] not in edge_raw]
    matched = [item for item in remaining if item["sig"] in edge_shear]
    unmatched = [item for item in remaining if item["sig"] not in edge_shear]
    shear_only = [item for item in matched if item["sig"] not in edge_uns]
    false_positive = [item for item in unmatched if item["sig"] in edge_uns]
    dihedral_only = [item for item in matched if item["tight_key"] not in edge_shear_keys]

    if not shear_only or not false_positive or not dihedral_only:
        return None

    for _ in range(60):
        match_count = random.choice([2, 3])
        used_raw: set[tuple[str, ...]] = set()
        inner: list[dict] = []

        both = [item for item in shear_only if item["tight_key"] not in edge_shear_keys]
        seed_pool = both if both else shear_only
        seed = _choose_one(seed_pool, used_raw)
        if seed is None:
            continue
        inner.append(seed)

        if seed["tight_key"] in edge_shear_keys:
            extra_green = _choose_one(dihedral_only, used_raw)
            if extra_green is None:
                continue
            inner.append(extra_green)

        more_green = match_count - len(inner)
        picked_green = _choose_many(matched, more_green, used_raw)
        if picked_green is None:
            continue
        inner.extend(picked_green)

        red_seed = _choose_one(false_positive, used_raw)
        if red_seed is None:
            continue
        inner.append(red_seed)

        more_red = len(INNER_POSITIONS) - len(inner)
        picked_red = _choose_many(unmatched, more_red, used_raw)
        if picked_red is None:
            continue
        inner.extend(picked_red)

        if len(inner) != len(INNER_POSITIONS):
            continue
        random.shuffle(inner)
        return inner
    return None


def _build_grid(edge_items: list[dict], inner_items: list[dict]) -> np.ndarray:
    grid = np.full(GRID_SHAPE, BG, dtype=int)
    for i in range(0, GRID_SHAPE[0], CELL_SIZE + 1):
        grid[i, :] = LINE
        grid[:, i] = LINE

    edge_masks = edge_items[:]
    inner_masks = inner_items[:]
    random.shuffle(edge_masks)
    random.shuffle(inner_masks)

    assignments: dict[tuple[int, int], dict] = {}
    for pos, item in zip(EDGE_POSITIONS, edge_masks):
        assignments[pos] = item
    for pos, item in zip(INNER_POSITIONS, inner_masks):
        assignments[pos] = item

    for (ri, ci), item in assignments.items():
        r0 = 1 + ri * (CELL_SIZE + 1)
        c0 = 1 + ci * (CELL_SIZE + 1)
        patch = grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
        patch[item["mask"]] = OBJ
    return grid


def _validate_sample(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.shape != GRID_SHAPE:
        return False

    analysis = _analyze_grid(grid)
    labels = [bool(cell["match"]) for cell in analysis["inner"]]
    no_shear = [bool(cell["no_shear_match"]) for cell in analysis["inner"]]

    if sum(labels) < 2 or sum(not flag for flag in labels) < 1:
        return False
    if labels == no_shear:
        return False
    if not any(cell["match"] and not cell["no_shear_match"] for cell in analysis["inner"]):
        return False
    if not any((not cell["match"]) and cell["no_shear_match"] for cell in analysis["inner"]):
        return False
    if not any(cell["match"] and not cell["exact_shear_match"] for cell in analysis["inner"]):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[grid == LINE], grid[grid == LINE]):
        return False

    for pos, mask in analysis["cells"].items():
        ri, ci = pos
        r0, r1 = analysis["row_spans"][ri]
        c0, c1 = analysis["col_spans"][ci]
        patch = crop_bbox(out, (r0, c0, r1, c1))
        if pos in EDGE_POSITIONS:
            if np.any(patch != BG):
                return False
            continue

        cell = next(item for item in analysis["inner"] if item["pos"] == pos)
        expected = MATCH if cell["match"] else MISS
        if np.any(patch[mask] != expected):
            return False
        if np.any(patch[~mask] != BG):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    edge_items = random.sample(CANDIDATES, len(EDGE_POSITIONS))
    inner_items = _choose_inner_masks(edge_items)
    if inner_items is None:
        return None
    grid = _build_grid(edge_items, inner_items)
    return grid if _validate_sample(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-shear lattice input")
