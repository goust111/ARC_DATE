from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
TRANSFORMS = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)
INVERSE = {
    "id": "id",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_lr": "flip_lr",
    "flip_ud": "flip_ud",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}

GRID_SIDE_RANGE = (16, 25)
PATTERN_SIDE_RANGE = (5, 8)
MAX_GENERATION_ATTEMPTS = 600


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _transform(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1).copy()
    if name == "rot180":
        return np.rot90(arr, 2).copy()
    if name == "rot270":
        return np.rot90(arr, 3).copy()
    if name == "flip_lr":
        return np.fliplr(arr).copy()
    if name == "flip_ud":
        return np.flipud(arr).copy()
    if name == "transpose":
        return arr.T.copy()
    if name == "anti_transpose":
        return np.fliplr(np.flipud(arr).T).copy()
    raise ValueError(f"unknown transform: {name}")


def _map_coord(r: int, c: int, n: int, name: str) -> tuple[int, int]:
    if name == "id":
        return r, c
    if name == "rot90":
        return n - 1 - c, r
    if name == "rot180":
        return n - 1 - r, n - 1 - c
    if name == "rot270":
        return c, n - 1 - r
    if name == "flip_lr":
        return r, n - 1 - c
    if name == "flip_ud":
        return n - 1 - r, c
    if name == "transpose":
        return c, r
    if name == "anti_transpose":
        return n - 1 - c, n - 1 - r
    raise ValueError(f"unknown transform: {name}")


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_box = _bbox(grid == FRAME)
    assert frame_box is not None, "expected one gray frame"
    r0, c0, r1, c1 = frame_box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must surround a non-empty interior"
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(frame[0, :] == FRAME)
    assert np.all(frame[-1, :] == FRAME)
    assert np.all(frame[:, 0] == FRAME)
    assert np.all(frame[:, -1] == FRAME)
    ring = np.zeros_like(grid, dtype=bool)
    ring[r0, c0 : c1 + 1] = True
    ring[r1, c0 : c1 + 1] = True
    ring[r0 : r1 + 1, c0] = True
    ring[r0 : r1 + 1, c1] = True
    assert np.array_equal(grid == FRAME, ring), "gray pixels must form exactly one rectangular frame"
    return r0, c0, r1, c1


def _pose_stats(partial: np.ndarray, fragment: np.ndarray) -> dict[str, dict[str, np.ndarray | bool]]:
    stats: dict[str, dict[str, np.ndarray | bool]] = {}
    for name in TRANSFORMS:
        transformed = _transform(fragment, name)
        if transformed.shape != partial.shape:
            continue
        overlap = (partial != BG) & (transformed != BG)
        conflict_free = bool(np.all(partial[overlap] == transformed[overlap]))
        fill_all = bool(np.all((partial != BG) | (transformed != BG)))
        merged = partial.copy()
        merged[merged == BG] = transformed[merged == BG]
        stats[name] = {
            "conflict_free": conflict_free,
            "fill_all": fill_all,
            "merged": merged,
        }
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full picture inside the frame by finding the only conflict-free D4 pose of the external complementary fragment.

    Input structure:
    - A black background contains exactly one gray rectangular frame.
    - Inside the frame is a square interior holding a partial multicolor pattern,
      with the missing cells shown as black.
    - Outside the frame sits another square patch of the same size containing the
      complementary fragment in a rotated or mirrored pose on black background.

    Rule:
    - Extract the frame interior as fragment A and the external square patch as
      fragment B.
    - Try every D4 transform of B (four rotations and four reflections).
    - Keep the unique pose whose colored overlaps with A never disagree and whose
      union with A fills every interior cell, then output that filled interior
      without the gray frame.

    Key insight:
    - Neither fragment is meaningful alone; the correct pose is determined only
      by the joint test "no color conflicts where they overlap" and "together
      they cover the whole framed square."
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    r0, c0, r1, c1 = _find_frame(g)

    partial = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    assert partial.shape[0] == partial.shape[1], "generator uses square interior so all D4 poses are available"

    outside_mask = (g != BG) & (g != FRAME)
    outside_mask[r0 + 1 : r1, c0 + 1 : c1] = False
    outside_box = _bbox(outside_mask)
    assert outside_box is not None, "expected one external fragment"
    fragment = _crop_bbox(g, outside_box)
    assert fragment.shape == partial.shape, "external fragment must match the frame interior"

    stats = _pose_stats(partial, fragment)
    valid = [name for name, info in stats.items() if info["conflict_free"] and info["fill_all"]]
    assert len(valid) == 1, "expected a unique valid pose"
    return np.asarray(stats[valid[0]]["merged"], dtype=int)


def _neighbors4(r: int, c: int, n: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < n:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < n:
        yield r, c + 1


def _add_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    if random.random() < 0.5:
        while c != ec:
            c += 1 if ec > c else -1
            mask[r, c] = True
        while r != er:
            r += 1 if er > r else -1
            mask[r, c] = True
    else:
        while r != er:
            r += 1 if er > r else -1
            mask[r, c] = True
        while c != ec:
            c += 1 if ec > c else -1
            mask[r, c] = True


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    n = mask.shape[0]
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc, n):
            if not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    n = mask.shape[0]
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, n):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(coords.shape[0])


def _edge_touch_count(mask: np.ndarray) -> int:
    return sum(
        (
            bool(mask[0, :].any()),
            bool(mask[-1, :].any()),
            bool(mask[:, 0].any()),
            bool(mask[:, -1].any()),
        )
    )


def _generate_hole_mask(n: int) -> np.ndarray | None:
    center = (random.randrange(n), random.randrange(n))
    anchors = [
        (0, random.randrange(n)),
        (n - 1, random.randrange(n)),
        (random.randrange(n), 0),
        (random.randrange(n), n - 1),
    ]
    mask = np.zeros((n, n), dtype=bool)
    for anchor in anchors:
        _add_path(mask, center, anchor)

    min_cells = max(int(np.ceil(0.34 * n * n)), int(mask.sum()))
    max_cells = int(np.floor(0.5 * n * n))
    if min_cells > max_cells:
        return None

    target = random.randint(min_cells, max_cells)
    while int(mask.sum()) < target:
        options = _frontier(mask)
        if not options:
            return None
        mask[random.choice(options)] = True

    if not _connected(mask):
        return None
    if _edge_touch_count(mask) != 4:
        return None
    if any(np.array_equal(mask, _transform(mask.astype(int), name).astype(bool)) for name in TRANSFORMS if name != "id"):
        return None
    return mask


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _idx(r: int, c: int, n: int) -> int:
    return r * n + c


def _build_equalities(bmask: np.ndarray, hole_mask: np.ndarray, no_conflict_name: str) -> _UnionFind:
    n = bmask.shape[0]
    uf = _UnionFind(n * n)
    for r, c in np.argwhere(bmask):
        rr = int(r)
        cc = int(c)
        tr, tc = _map_coord(rr, cc, n, no_conflict_name)
        if not hole_mask[tr, tc]:
            uf.union(_idx(rr, cc, n), _idx(tr, tc, n))
    return uf


def _component_map(uf: _UnionFind, n: int) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(n):
        for c in range(n):
            root = uf.find(_idx(r, c, n))
            groups.setdefault(root, []).append((r, c))
    return groups


def _assign_pattern(
    uf: _UnionFind,
    n: int,
    conflict_name: str,
    bmask: np.ndarray,
    hole_mask: np.ndarray,
) -> np.ndarray | None:
    witness_pairs: list[tuple[int, int]] = []
    for r, c in np.argwhere(bmask):
        rr = int(r)
        cc = int(c)
        tr, tc = _map_coord(rr, cc, n, conflict_name)
        if hole_mask[tr, tc]:
            continue
        a = uf.find(_idx(rr, cc, n))
        b = uf.find(_idx(tr, tc, n))
        if a != b:
            witness_pairs.append((a, b))
    if not witness_pairs:
        return None

    groups = _component_map(uf, n)
    roots = list(groups)
    random.shuffle(roots)
    n_colors = min(len(roots), random.randint(3, 5))
    palette = random.sample(PALETTE, n_colors)
    root_color = {root: random.choice(palette) for root in roots}
    wa, wb = random.choice(witness_pairs)
    root_color[wa] = palette[0]
    root_color[wb] = palette[1 if len(palette) > 1 else 0]
    if root_color[wa] == root_color[wb]:
        return None

    pattern = np.zeros((n, n), dtype=int)
    for root, cells in groups.items():
        color = int(root_color[root])
        for r, c in cells:
            pattern[r, c] = color

    if len(set(map(int, np.unique(pattern)))) < 3:
        return None
    if any(np.array_equal(pattern, _transform(pattern, name)) for name in TRANSFORMS if name != "id"):
        return None
    row_changes = int(np.sum(pattern[:, 1:] != pattern[:, :-1]))
    col_changes = int(np.sum(pattern[1:, :] != pattern[:-1, :]))
    if row_changes + col_changes < n * 3:
        return None
    return pattern


def _coverage(mask: np.ndarray, name: str) -> np.ndarray:
    return _transform(mask.astype(int), name).astype(bool)


def _semantic_ok(
    pattern: np.ndarray,
    hole_mask: np.ndarray,
    bmask: np.ndarray,
    display_name: str,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    partial = pattern.copy()
    partial[hole_mask] = BG

    fragment = np.zeros_like(pattern)
    fragment[bmask] = pattern[bmask]
    stats = _pose_stats(partial, fragment)

    valid = [name for name, info in stats.items() if info["conflict_free"] and info["fill_all"]]
    if valid != ["id"]:
        return False, None, None

    wrong_fill_conflict = [
        name
        for name, info in stats.items()
        if name != "id" and info["fill_all"] and not info["conflict_free"]
    ]
    wrong_safe_incomplete = [
        name
        for name, info in stats.items()
        if name != "id" and info["conflict_free"] and not info["fill_all"]
    ]
    if display_name not in wrong_fill_conflict:
        return False, None, None
    if not wrong_safe_incomplete:
        return False, None, None

    displayed = _transform(fragment, display_name)
    raw_stats = _pose_stats(partial, displayed)
    solver_name = INVERSE[display_name]
    valid_raw = [name for name, info in raw_stats.items() if info["conflict_free"] and info["fill_all"]]
    if valid_raw != [solver_name]:
        return False, None, None
    if bool(raw_stats["id"]["conflict_free"]):
        return False, None, None
    return True, partial, displayed


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _render_input(partial: np.ndarray, displayed: np.ndarray) -> np.ndarray | None:
    n = partial.shape[0]
    frame_side = n + 2
    rows = random.randint(max(frame_side, n) + 4, GRID_SIDE_RANGE[1])
    cols = random.randint(max(frame_side, n) + 4, GRID_SIDE_RANGE[1])
    grid = np.zeros((rows, cols), dtype=int)

    for _ in range(80):
        fr = random.randint(1, rows - frame_side - 1)
        fc = random.randint(1, cols - frame_side - 1)
        br = random.randint(1, rows - n - 1)
        bc = random.randint(1, cols - n - 1)
        frame_box = (fr, fc, fr + frame_side - 1, fc + frame_side - 1)
        patch_box = (br, bc, br + n - 1, bc + n - 1)
        if not _boxes_separated(frame_box, patch_box, gap=1):
            continue

        grid.fill(BG)
        grid[fr, fc : fc + frame_side] = FRAME
        grid[fr + frame_side - 1, fc : fc + frame_side] = FRAME
        grid[fr : fr + frame_side, fc] = FRAME
        grid[fr : fr + frame_side, fc + frame_side - 1] = FRAME
        _paste(grid, partial, fr + 1, fc + 1, transparent=BG)
        _paste(grid, displayed, br, bc, transparent=BG)
        return grid
    return None


def _attempt() -> np.ndarray | None:
    n = random.randint(*PATTERN_SIDE_RANGE)
    hole_mask = _generate_hole_mask(n)
    if hole_mask is None:
        return None

    display_name = random.choice(TRANSFORMS[1:])
    transformed_holes = _coverage(hole_mask, display_name)
    missing_after_wrong = hole_mask & ~transformed_holes
    if not missing_after_wrong.any():
        return None

    overlap_mask = _coverage(missing_after_wrong, INVERSE[display_name])
    if np.any(overlap_mask & hole_mask):
        return None

    bmask = hole_mask | overlap_mask
    if int(bmask.sum()) >= int(0.72 * n * n):
        return None
    if int(bmask.sum()) <= int(0.42 * n * n):
        return None
    if _edge_touch_count(bmask) != 4:
        return None

    no_conflict_candidates = list(TRANSFORMS[1:])
    random.shuffle(no_conflict_candidates)
    for no_conflict_name in no_conflict_candidates:
        if no_conflict_name == display_name:
            continue
        if np.all(_coverage(bmask, no_conflict_name) >= hole_mask):
            continue

        uf = _build_equalities(bmask, hole_mask, no_conflict_name)
        for _ in range(20):
            pattern = _assign_pattern(uf, n, display_name, bmask, hole_mask)
            if pattern is None:
                continue
            ok, partial, displayed = _semantic_ok(pattern, hole_mask, bmask, display_name)
            if not ok:
                continue
            grid = _render_input(partial, displayed)
            if grid is None:
                continue
            if not np.array_equal(main(grid), pattern):
                continue
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment ARC input")
