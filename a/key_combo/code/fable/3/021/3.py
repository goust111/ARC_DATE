from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
COLORS = list(range(1, 10))


SHAPES: dict[str, frozenset[tuple[int, int]]] = {
    "DOT": frozenset({(0, 0)}),
    "H2": frozenset({(0, 0), (0, 1)}),
    "V2": frozenset({(0, 0), (1, 0)}),
    "L_UL": frozenset({(0, 0), (0, 1), (1, 0)}),
    "L_UR": frozenset({(0, 0), (0, 1), (1, 1)}),
    "L_BL": frozenset({(0, 0), (1, 0), (1, 1)}),
    "L_BR": frozenset({(0, 1), (1, 0), (1, 1)}),
}
SHAPE_BY_CELLS = {cells: name for name, cells in SHAPES.items()}
L_SHAPES = ["L_UL", "L_UR", "L_BL", "L_BR"]
STRAIGHT_SHAPES = ["DOT", "H2", "V2"]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Connected components as (N,2) arrays, using the arc_utils (r,c) convention."""
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 8:
        dirs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        out.append(np.array(cells, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_cells(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells)


def _shape_name(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> str | None:
    return SHAPE_BY_CELLS.get(_norm_cells(cells))


def _ideal_border(b: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = b
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _local_l_gap(shape: str, h: int, w: int) -> set[tuple[int, int]]:
    if shape == "L_UL":
        return {(0, 0), (0, 1), (1, 0)}
    if shape == "L_UR":
        return {(0, w - 2), (0, w - 1), (1, w - 1)}
    if shape == "L_BL":
        return {(h - 2, 0), (h - 1, 0), (h - 1, 1)}
    if shape == "L_BR":
        return {(h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)}
    raise ValueError(shape)


def _gap_shape(missing: set[tuple[int, int]], b: tuple[int, int, int, int]) -> str | None:
    if not missing:
        return None
    name = _shape_name(missing)
    if name is None:
        return None
    if name not in L_SHAPES:
        return name

    r0, c0, r1, c1 = b
    h, w = r1 - r0 + 1, c1 - c0 + 1
    for lname in L_SHAPES:
        expected = {(r0 + r, c0 + c) for r, c in _local_l_gap(lname, h, w)}
        if missing == expected:
            return lname
    return None


def _frame_info(coords: np.ndarray) -> dict | None:
    if coords.shape[0] < 7:
        return None
    b = _bbox_coords(coords)
    r0, c0, r1, c1 = b
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    cells = {(int(r), int(c)) for r, c in coords}
    border = _ideal_border(b)
    if not cells <= border:
        return None
    missing = border - cells
    name = _gap_shape(missing, b)
    if name is None:
        return None
    if cells != border - missing:
        return None
    return {"bbox": b, "gap": missing, "shape": name}


def _detect_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    candidates: dict[int, list[dict]] = {}
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        infos = []
        for coords in _components(grid == color, connectivity=4):
            info = _frame_info(coords)
            if info is not None:
                info = dict(info)
                info["color"] = int(color)
                infos.append(info)
        if infos:
            candidates[int(color)] = infos

    assert candidates, "expected at least one incomplete hollow rectangle"
    frame_color = max(
        candidates,
        key=lambda col: (
            len(candidates[col]),
            sum(len(info["gap"]) + len(_ideal_border(info["bbox"])) for info in candidates[col]),
        ),
    )
    return int(frame_color), candidates[frame_color]


def _key_components(grid: np.ndarray, frame_color: int) -> list[dict]:
    keys: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, int(frame_color))):
        for coords in _components(grid == color, connectivity=4):
            cells = {(int(r), int(c)) for r, c in coords}
            keys.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "cells": cells,
                    "shape": _shape_name(cells),
                }
            )
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each small key shape into the only complementary notch in a hollow frame.

    Input structure:
    - A black-background grid contains several one-cell-thick hollow rectangular
      frames in one shared frame color.
    - Each frame has one missing wall notch: a single cell, a two-cell straight
      notch, or a three-cell L notch at a corner.
    - Separate monochrome key components use other colors; some key shapes match
      exactly one notch by translation only, while extra keys and unpaired frames
      may also be present.

    Rule:
    - Erase all original key components.
    - For every notch shape that appears on exactly one frame and exactly one key,
      translate that key into the notch without rotation or reflection.
    - A frame closed by a fitted key has its strict rectangular interior filled
      with the fitted key's color; frames with no unique fitted key stay open and
      their interiors remain black.

    Key insight:
    - The fill color is not taken from the frame. Shape complement matching first
      closes a specific wall gap, and only that closure triggers the interior fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color, frames = _detect_frames(g)
    keys = _key_components(g, frame_color)

    gap_counts = Counter(info["shape"] for info in frames)
    keys_by_shape: dict[str, list[dict]] = defaultdict(list)
    for key in keys:
        if key["shape"] is not None:
            keys_by_shape[key["shape"]].append(key)

    out = g.copy()
    out[(out != BG) & (out != frame_color)] = BG

    for frame in frames:
        shape = frame["shape"]
        key_list = keys_by_shape.get(shape, [])
        if gap_counts[shape] != 1 or len(key_list) != 1:
            continue
        key_color = int(key_list[0]["color"])
        for r, c in frame["gap"]:
            out[int(r), int(c)] = key_color

        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = key_color

    return out


def _shape_patch(shape: str, color: int) -> np.ndarray:
    cells = SHAPES[shape]
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    patch = np.full((h, w), BG, dtype=int)
    for r, c in cells:
        patch[int(r), int(c)] = int(color)
    return patch


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _box_free(occupied: np.ndarray, h: int, w: int, r0: int, c0: int, pad: int, margin: int) -> bool:
    H, W = occupied.shape
    if r0 < margin or c0 < margin or r0 + h > H - margin or c0 + w > W - margin:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _reserve_box(occupied: np.ndarray, h: int, w: int, r0: int, c0: int, pad: int) -> None:
    H, W = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _random_box_location(
    occupied: np.ndarray, h: int, w: int, pad: int = 1, margin: int = 1
) -> tuple[int, int] | None:
    H, W = occupied.shape
    candidates = [
        (r, c)
        for r in range(0, H - h + 1)
        for c in range(0, W - w + 1)
        if _box_free(occupied, h, w, r, c, pad=pad, margin=margin)
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _random_local_gap(shape: str, h: int, w: int) -> set[tuple[int, int]]:
    if shape == "DOT":
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            return {(0, random.randint(1, w - 2))}
        if side == "bottom":
            return {(h - 1, random.randint(1, w - 2))}
        if side == "left":
            return {(random.randint(1, h - 2), 0)}
        return {(random.randint(1, h - 2), w - 1)}

    if shape == "H2":
        row = random.choice([0, h - 1])
        start = random.randint(1, w - 3)
        return {(row, start), (row, start + 1)}

    if shape == "V2":
        col = random.choice([0, w - 1])
        start = random.randint(1, h - 3)
        return {(start, col), (start + 1, col)}

    return _local_l_gap(shape, h, w)


def _frame_patch(h: int, w: int, shape: str, color: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    gap = _random_local_gap(shape, h, w)
    for r, c in gap:
        patch[int(r), int(c)] = BG
    return patch, gap


def _choose_frame_shapes(n_frames: int) -> list[str]:
    shapes = [random.choice(L_SHAPES), random.choice(["H2", "V2"]), "DOT"]
    pool = [s for s in SHAPES if s not in shapes]
    random.shuffle(pool)
    shapes.extend(pool[: max(0, n_frames - len(shapes))])
    random.shuffle(shapes)
    return shapes[:n_frames]


def _choose_matched_shapes(frame_shapes: list[str], n_matched: int) -> set[str]:
    matched = {random.choice([s for s in frame_shapes if s in L_SHAPES])}
    non_l = [s for s in frame_shapes if s not in L_SHAPES]
    if non_l:
        matched.add(random.choice(non_l))
    rest = [s for s in frame_shapes if s not in matched]
    random.shuffle(rest)
    for s in rest:
        if len(matched) >= n_matched:
            break
        matched.add(s)
    return matched


def _render_expected(
    grid: np.ndarray,
    frames: list[dict],
    key_records: list[dict],
    matched_colors: dict[str, int],
) -> np.ndarray:
    out = grid.copy()
    for key in key_records:
        for r, c in key["source_cells"]:
            out[int(r), int(c)] = BG

    for frame in frames:
        shape = frame["shape"]
        if shape not in matched_colors:
            continue
        color = int(matched_colors[shape])
        for r, c in frame["gap_abs"]:
            out[int(r), int(c)] = color
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    frames: list[dict],
    key_records: list[dict],
    matched_colors: dict[str, int],
    extra_shapes: set[str],
) -> bool:
    frame_shapes = [f["shape"] for f in frames]
    if len(frame_shapes) != len(set(frame_shapes)):
        return False
    if not matched_colors or len(matched_colors) >= len(frames):
        return False
    if not extra_shapes or extra_shapes & set(frame_shapes):
        return False
    if not any(s in L_SHAPES for s in matched_colors):
        return False
    if not any(s in STRAIGHT_SHAPES for s in matched_colors):
        return False

    for key in key_records:
        for r, c in key["source_cells"]:
            if int(expected[int(r), int(c)]) != BG:
                return False

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = expected[r0 + 1 : r1, c0 + 1 : c1]
        if frame["shape"] in matched_colors:
            color = int(matched_colors[frame["shape"]])
            if any(int(expected[int(r), int(c)]) != color for r, c in frame["gap_abs"]):
                return False
            if not bool(np.all(interior == color)):
                return False
        else:
            if any(int(expected[int(r), int(c)]) != BG for r, c in frame["gap_abs"]):
                return False
            if not bool(np.all(interior == BG)):
                return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(20, 30)
    W = random.randint(20, 30)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    frame_color = random.choice(COLORS)
    n_frames = random.randint(3, 4)
    frame_shapes = _choose_frame_shapes(n_frames)
    n_matched = random.randint(2, n_frames - 1)
    matched_shapes = _choose_matched_shapes(frame_shapes, n_matched)

    frames: list[dict] = []
    for shape in frame_shapes:
        h = random.randint(5, 8)
        w = random.randint(5, 9)
        patch, local_gap = _frame_patch(h, w, shape, frame_color)
        loc = _random_box_location(occupied, h, w, pad=1, margin=1)
        if loc is None:
            return None
        r0, c0 = loc
        _paste_patch(grid, patch, r0, c0)
        _reserve_box(occupied, h, w, r0, c0, pad=1)
        frames.append(
            {
                "shape": shape,
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                "gap_abs": {(r0 + r, c0 + c) for r, c in local_gap},
            }
        )

    absent_shapes = [s for s in SHAPES if s not in frame_shapes]
    n_extra = random.randint(1, min(2, len(absent_shapes)))
    extra_shapes = set(random.sample(absent_shapes, n_extra))

    key_specs = [("matched", s) for s in matched_shapes] + [("extra", s) for s in extra_shapes]
    random.shuffle(key_specs)
    key_colors = random.sample([c for c in COLORS if c != frame_color], len(key_specs))

    key_records: list[dict] = []
    matched_colors: dict[str, int] = {}
    for (kind, shape), color in zip(key_specs, key_colors):
        patch = _shape_patch(shape, color)
        h, w = patch.shape
        loc = _random_box_location(occupied, h, w, pad=1, margin=1)
        if loc is None:
            return None
        r0, c0 = loc
        _paste_patch(grid, patch, r0, c0)
        _reserve_box(occupied, h, w, r0, c0, pad=1)
        source_cells = {(r0 + r, c0 + c) for r, c in SHAPES[shape]}
        key_records.append({"shape": shape, "source_cells": source_cells, "color": int(color)})
        if kind == "matched":
            matched_colors[shape] = int(color)

    expected = _render_expected(grid, frames, key_records, matched_colors)
    if not _semantic_ok(grid, expected, frames, key_records, matched_colors, extra_shapes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock puzzle input")
