from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATION_RETRIES = 400

SHAPES: dict[str, np.ndarray] = {
    "domino": np.array([[1, 1]], dtype=bool),
    "line3": np.array([[1, 1, 1]], dtype=bool),
    "el3": np.array([[1, 0], [1, 1]], dtype=bool),
}


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _same_color_component(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    sr, sc = start
    color = int(grid[sr, sc])
    seen = {start}
    stack = [start]
    coords: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return np.array(coords, dtype=int)


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            trimmed = _trim(cand)
            key = (trimmed.shape, trimmed.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(trimmed)
    return variants


def _canonical(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    for variant in _unique_variants(mask):
        key = (variant.shape, variant.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _mask_from_coords(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    pts = list(coords)
    assert pts
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    out = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for r, c in pts:
        out[r - min_r, c - min_c] = True
    return out


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _row_runs(grid: np.ndarray, color: int, min_len: int = 1) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    runs: list[tuple[int, int, int]] = []
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != color:
                c += 1
                continue
            c0 = c
            while c < w and int(grid[r, c]) == color:
                c += 1
            c1 = c - 1
            if c1 - c0 + 1 >= min_len:
                runs.append((r, c0, c1))
    return runs


def _col_runs(grid: np.ndarray, color: int, min_len: int = 1) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    runs: list[tuple[int, int, int]] = []
    for c in range(w):
        r = 0
        while r < h:
            if int(grid[r, c]) != color:
                r += 1
                continue
            r0 = r
            while r < h and int(grid[r, c]) == color:
                r += 1
            r1 = r - 1
            if r1 - r0 + 1 >= min_len:
                runs.append((c, r0, r1))
    return runs


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    for color in colors:
        vertical_index: dict[tuple[int, int], list[int]] = {}
        for c, r0, r1 in _col_runs(grid, color, min_len=5):
            vertical_index.setdefault((r1, c), []).append(r0)
        for r1, c0, c1 in _row_runs(grid, color, min_len=5):
            for r0 in vertical_index.get((r1, c1), []):
                if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
                    continue
                if int(grid[r0, c0]) != color or int(grid[r1, c0]) != color:
                    continue
                top = grid[r0, c0 : c1 + 1]
                left = grid[r0 : r1 + 1, c0]
                if np.any(top == bg) or np.any(left == bg):
                    continue
                if not (np.any(top != color) or np.any(left != color)):
                    continue
                interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                if interior.size == 0 or not np.any(interior == bg):
                    continue
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                candidates.append((area, color, (r0, c0, r1, c1)))
    assert candidates, "frame not found"
    _, color, bbox = max(candidates, key=lambda item: item[0])
    return color, bbox


def _extract_key_map(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
) -> dict[tuple[tuple[int, int], bytes], int]:
    r0, c0, r1, c1 = frame_bbox
    labels: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        if int(grid[r0, c]) not in (bg, frame_color):
            labels.append((r0, c))
    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) not in (bg, frame_color):
            labels.append((r, c0))
    assert labels, "no labels found"

    key_map: dict[tuple[tuple[int, int], bytes], int] = {}
    seen: set[tuple[int, int]] = set()
    for label in labels:
        if label in seen:
            continue
        comp = _same_color_component(grid, label)
        for cell in map(tuple, comp.tolist()):
            seen.add(cell)
        border_cells = [
            (r, c)
            for r, c in map(tuple, comp.tolist())
            if ((r == r0 or r == r1) and c0 <= c <= c1) or ((c == c0 or c == c1) and r0 <= r <= r1)
        ]
        assert len(border_cells) == 1, "legend component must touch the frame at exactly one label"
        lr, lc = border_cells[0]
        assert lr == r0 or lc == c0, "labels must sit on the top or left border"
        key_pts = [(r, c) for r, c in map(tuple, comp.tolist()) if (r, c) != (lr, lc)]
        assert key_pts, "missing outside key pixels"
        assert all(r < r0 or c < c0 for r, c in key_pts), "key must stay outside the frame"
        shape_key = _canonical(_mask_from_coords(key_pts))
        color = int(grid[lr, lc])
        assert shape_key not in key_map, "duplicate key shape"
        key_map[shape_key] = color
    return key_map


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    key_map = _extract_key_map(g, bg, frame_color, frame_bbox)

    r0, c0, r1, c1 = frame_bbox
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    object_components = _components(interior == frame_color)
    assert object_components, "no interior objects found"

    out = g.copy()
    for comp in object_components:
        rr0, cc0, rr1, cc1 = _bbox(comp)
        patch = interior[rr0 : rr1 + 1, cc0 : cc1 + 1]
        assert np.all((patch == frame_color) | (patch == bg)), "object patch must only contain frame color and background"
        assert np.all(patch[0, :] == frame_color)
        assert np.all(patch[-1, :] == frame_color)
        assert np.all(patch[:, 0] == frame_color)
        assert np.all(patch[:, -1] == frame_color)

        hole_comps = [hole for hole in _components(patch == bg) if not _touches_boundary(hole, patch.shape)]
        assert len(hole_comps) == 1, "each object must have exactly one enclosed hole"
        hole = hole_comps[0]
        hole_mask = np.zeros(patch.shape, dtype=bool)
        hole_mask[hole[:, 0], hole[:, 1]] = True
        shape_key = _canonical(_trim(hole_mask))
        color = key_map[shape_key]

        global_rows = comp[:, 0] + (r0 + 1)
        global_cols = comp[:, 1] + (c0 + 1)
        out[global_rows, global_cols] = color

    return out


def _side_placements(mask: np.ndarray, side: str) -> list[tuple[tuple[int, int], ...]]:
    placements: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for variant in _unique_variants(mask):
        coords = np.argwhere(variant)
        if side == "top":
            anchors = coords[coords[:, 0] == coords[:, 0].max()]
        else:
            anchors = coords[coords[:, 1] == coords[:, 1].max()]
        for ar, ac in anchors:
            rel = tuple(sorted((int(r - ar), int(c - ac)) for r, c in coords))
            if side == "top" and any(dr > 0 for dr, _ in rel):
                continue
            if side == "left" and any(dc > 0 for _, dc in rel):
                continue
            if rel not in seen:
                seen.add(rel)
                placements.append(rel)
    return placements


SIDE_CACHE = {name: {"top": _side_placements(mask, "top"), "left": _side_placements(mask, "left")} for name, mask in SHAPES.items()}


def _make_object_patch(shape_name: str) -> tuple[np.ndarray, np.ndarray]:
    hole_mask = random.choice(_unique_variants(SHAPES[shape_name]))
    hh, hw = hole_mask.shape
    obj_h = random.randint(max(4, hh + 2), max(6, hh + 2))
    obj_w = random.randint(max(4, hw + 2), max(6, hw + 2))
    patch = np.full((obj_h, obj_w), FRAME, dtype=int)
    hole_r = random.randint(1, obj_h - hh - 1)
    hole_c = random.randint(1, obj_w - hw - 1)
    patch[hole_r : hole_r + hh, hole_c : hole_c + hw][hole_mask] = BG
    body_mask = patch == FRAME
    return patch, body_mask


def _place_objects(inner_h: int, inner_w: int, patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        ph, pw = patches[idx].shape
        options: list[tuple[int, int]] = []
        for r in range(1, inner_h - ph):
            for c in range(1, inner_w - pw):
                rr0 = r - 1
                cc0 = c - 1
                rr1 = r + ph + 1
                cc1 = c + pw + 1
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                options.append((r, c))
        if not options:
            return None
        r, c = random.choice(options)
        placements[idx] = (r, c)
        occupied[r - 1 : r + ph + 1, c - 1 : c + pw + 1] = True

    return [pos for pos in placements if pos is not None]


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _try_place_key(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    color: int,
    shape_name: str,
    used_positions: set[int],
) -> bool:
    r0, c0, r1, c1 = frame_bbox
    placements = SIDE_CACHE[shape_name][side][:]
    random.shuffle(placements)
    if side == "top":
        candidates = list(range(c0 + 2, c1 - 1))
    else:
        candidates = list(range(r0 + 2, r1 - 1))
    random.shuffle(candidates)

    for rel in placements:
        for pos in candidates:
            if any(abs(pos - other) < 2 for other in used_positions):
                continue
            if side == "top":
                label = (r0, pos)
                key_cells = [(r0 - 1 + dr, pos + dc) for dr, dc in rel]
            else:
                label = (pos, c0)
                key_cells = [(pos + dr, c0 - 1 + dc) for dr, dc in rel]
            if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in key_cells):
                continue
            if int(grid[label]) != FRAME:
                continue
            if any(int(grid[r, c]) != BG for r, c in key_cells):
                continue
            grid[label] = color
            for r, c in key_cells:
                grid[r, c] = color
            used_positions.add(pos)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n_keys = random.randint(2, 3)
    shape_names = random.sample(list(SHAPES), n_keys)
    colors = random.sample(PALETTE, n_keys)
    shape_to_color = dict(zip(shape_names, colors))

    top_count = random.randint(1, n_keys - 1)
    top_shapes = set(random.sample(shape_names, top_count))
    left_shapes = [name for name in shape_names if name not in top_shapes]

    n_objects = random.randint(n_keys + 1, n_keys + 4)
    object_shapes = shape_names[:] + [random.choice(shape_names) for _ in range(n_objects - n_keys)]
    random.shuffle(object_shapes)

    patches: list[np.ndarray] = []
    body_masks: list[np.ndarray] = []
    for shape_name in object_shapes:
        patch, body_mask = _make_object_patch(shape_name)
        patches.append(patch)
        body_masks.append(body_mask)

    inner_h = random.randint(13, 18)
    inner_w = random.randint(13, 18)
    placements = _place_objects(inner_h, inner_w, patches)
    if placements is None:
        return None

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    top_margin = random.randint(3, 5)
    left_margin = random.randint(3, 5)
    bottom_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    grid_h = top_margin + frame_h + bottom_margin
    grid_w = left_margin + frame_w + right_margin
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    frame_bbox = (r0, c0, r1, c1)
    _paint_frame(grid, r0, c0, r1, c1, FRAME)

    for idx, (patch, body_mask, (pr, pc), shape_name) in enumerate(zip(patches, body_masks, placements, object_shapes)):
        ph, pw = patch.shape
        rr = r0 + 1 + pr
        cc = c0 + 1 + pc
        region = grid[rr : rr + ph, cc : cc + pw]
        if np.any(region != BG):
            return None
        region[:] = patch

    top_used: set[int] = set()
    left_used: set[int] = set()
    for shape_name in random.sample(list(top_shapes), len(top_shapes)):
        if not _try_place_key(grid, frame_bbox, "top", shape_to_color[shape_name], shape_name, top_used):
            return None
    for shape_name in random.sample(left_shapes, len(left_shapes)):
        if not _try_place_key(grid, frame_bbox, "left", shape_to_color[shape_name], shape_name, left_used):
            return None

    if not top_used or not left_used:
        return None
    if len(np.unique(grid)) < 4:
        return None
    expected = grid.copy()
    for patch, body_mask, (pr, pc), shape_name in zip(patches, body_masks, placements, object_shapes):
        ph, pw = patch.shape
        rr = r0 + 1 + pr
        cc = c0 + 1 + pc
        expected_region = expected[rr : rr + ph, cc : cc + pw]
        expected_region[body_mask] = shape_to_color[shape_name]
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
