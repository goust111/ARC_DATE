from __future__ import annotations

import math
import random
from collections import Counter, deque

import numpy as np


Tree = tuple[str, tuple["Tree", ...]]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _tight_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((1, 1), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_mask(component_mask: np.ndarray) -> np.ndarray:
    h, w = component_mask.shape
    empty = ~component_mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return empty & ~outside


def _coords_inside_local_mask(
    coords: np.ndarray, bbox: tuple[int, int, int, int], local_mask: np.ndarray
) -> bool:
    r0, c0, _r1, _c1 = bbox
    rr = coords[:, 0] - int(r0)
    cc = coords[:, 1] - int(c0)
    h, w = local_mask.shape
    if np.any(rr < 0) or np.any(cc < 0) or np.any(rr >= h) or np.any(cc >= w):
        return False
    return bool(np.all(local_mask[rr, cc]))


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    fg_mask = np.asarray(grid != bg, dtype=bool)
    components = _components_mask(fg_mask, connectivity=4)
    records: list[dict] = []
    for coords in components:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
        holes = _hole_mask(patch)
        records.append(
            {
                "coords": coords,
                "bbox": bbox,
                "hole_mask": holes,
                "is_frame": bool(holes.any()),
                "children": [],
                "parent": None,
            }
        )

    frame_ids = [i for i, rec in enumerate(records) if rec["is_frame"]]
    for idx, rec in enumerate(records):
        best_parent = None
        best_area = None
        for frame_id in frame_ids:
            if frame_id == idx:
                continue
            frame = records[frame_id]
            if _coords_inside_local_mask(rec["coords"], frame["bbox"], frame["hole_mask"]):
                r0, c0, r1, c1 = frame["bbox"]
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if best_area is None or area < best_area:
                    best_parent = frame_id
                    best_area = area
        rec["parent"] = best_parent

    for idx, rec in enumerate(records):
        parent = rec["parent"]
        if parent is not None:
            records[parent]["children"].append(idx)

    return records


def _tree_signature(root: int, records: list[dict]) -> tuple:
    child_sigs = tuple(sorted(_tree_signature(child, records) for child in records[root]["children"]))
    if records[root]["is_frame"]:
        return ("F", child_sigs)
    return ("B", ())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely different nested containment topology and crop that whole structure.

    Input structure:
    - The grid has exactly one dominant background color and one foreground color.
    - Foreground pixels form several disjoint structures made of closed frames and solid blobs.
    - Frames may be skewed or irregular, and children can be nested inside other frames.

    Rule:
    - Split the foreground into connected components and treat every closed loop as a frame whose enclosed region may contain child components.
    - Build a containment tree for each top-level structure, ignoring the exact shape, size, and orientation of every frame or blob.
    - All but one top-level structure share the same tree up to isomorphism; crop and return the unique outlier structure in its original orientation.

    Key insight:
    - Geometry is a distraction: the identity of each structure is the unordered tree induced by which components are enclosed inside which frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    records = _component_records(g, bg)
    assert records

    roots = [idx for idx, rec in enumerate(records) if rec["parent"] is None]
    assert len(roots) >= 2
    signatures = {root: _tree_signature(root, records) for root in roots}
    counts = Counter(signatures.values())
    assert len(counts) == 2
    unique_sigs = [sig for sig, n in counts.items() if n == 1]
    assert len(unique_sigs) == 1
    outlier_root = next(root for root in roots if signatures[root] == unique_sigs[0])
    return _crop_bbox(g, records[outlier_root]["bbox"])


def _blob() -> Tree:
    return ("B", ())


def _frame(*children: Tree) -> Tree:
    return ("F", tuple(children))


TREE_LIBRARY: tuple[Tree, ...] = (
    _frame(_blob()),
    _frame(_frame(_blob())),
    _frame(_blob(), _blob()),
    _frame(_frame(), _blob()),
    _frame(_frame(_blob()), _blob()),
    _frame(_frame(), _frame(_blob())),
    _frame(_frame(_blob(), _blob())),
    _frame(_blob(), _frame(_blob())),
    _frame(_frame(), _frame()),
    _frame(_blob(), _blob(), _blob()),
    _frame(_frame(), _blob(), _blob()),
)

CHALLENGE_PAIRS: tuple[tuple[Tree, Tree], ...] = (
    (_frame(_frame(_blob())), _frame(_frame(), _blob())),
    (_frame(_frame(_blob()), _blob()), _frame(_frame(_blob(), _blob()))),
    (_frame(_frame(), _frame(_blob())), _frame(_blob(), _frame(_blob()))),
)

BLOB_SHAPES: tuple[np.ndarray, ...] = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
)

MAX_GRID_SIDE = 30
MAX_STRUCTURE_SIDE = 18
MAX_GENERATION_ATTEMPTS = 400
MAX_PATCH_RENDER_ATTEMPTS = 12


def _tree_signature_from_template(tree: Tree) -> tuple:
    kind, children = tree
    child_sigs = tuple(sorted(_tree_signature_from_template(child) for child in children))
    if kind == "F":
        return ("F", child_sigs)
    return ("B", ())


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k)
    if random.random() < 0.5:
        out = out[::-1, :]
    if random.random() < 0.5:
        out = out[:, ::-1]
    return _tight_bool(out)


def _render_blob_patch() -> np.ndarray:
    return _random_transform(random.choice(BLOB_SHAPES))


def _rectangles_separated(
    r: int, c: int, h: int, w: int, placed: list[tuple[int, int, int, int]], gap: int
) -> bool:
    for pr, pc, ph, pw in placed:
        if not (r + h - 1 + gap < pr or pr + ph - 1 + gap < r or c + w - 1 + gap < pc or pc + pw - 1 + gap < c):
            return False
    return True


def _pack_rectangles(
    sizes: list[tuple[int, int]],
    *,
    gap: int,
    min_h: int,
    min_w: int,
    max_h: int,
    max_w: int,
    tries_per_canvas: int = 25,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    if not sizes:
        return random.randint(min_h, min(min_h + 2, max_h)), random.randint(min_w, min(min_w + 2, max_w)), []

    lower_h = max(min_h, max(h for h, _w in sizes))
    lower_w = max(min_w, max(w for _h, w in sizes))
    inflated_area = sum((h + gap) * (w + gap) for h, w in sizes)
    candidates: list[tuple[int, int]] = []
    for h in range(lower_h, max_h + 1):
        for w in range(lower_w, max_w + 1):
            if h * w >= inflated_area - gap * max(1, len(sizes) - 1):
                candidates.append((h, w))
    if not candidates:
        return None

    random.shuffle(candidates)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for canvas_h, canvas_w in candidates[:140]:
        for _ in range(tries_per_canvas):
            placed: list[tuple[int, int, int, int]] = []
            positions: list[tuple[int, int] | None] = [None] * len(sizes)
            ok = True
            for idx in order:
                h, w = sizes[idx]
                options: list[tuple[int, int]] = []
                for r in range(canvas_h - h + 1):
                    for c in range(canvas_w - w + 1):
                        if _rectangles_separated(r, c, h, w, placed, gap):
                            options.append((r, c))
                if not options:
                    ok = False
                    break
                r, c = random.choice(options)
                positions[idx] = (r, c)
                placed.append((r, c, h, w))
            if ok:
                return canvas_h, canvas_w, [pos for pos in positions if pos is not None]
    return None


def _add_frame_protrusions(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    canvas = np.zeros((h + 2, w + 2), dtype=bool)
    canvas[1 : h + 1, 1 : w + 1] = mask
    sides = ["T", "B", "L", "R"]
    random.shuffle(sides)
    bump_count = random.randint(0, 3)
    if bump_count == 0 and max(h, w) >= 7 and random.random() < 0.7:
        bump_count = 1

    for side in sides[:bump_count]:
        if side in ("T", "B"):
            span_max = max(1, min(w, max(1, w // 2)))
            span = random.randint(1, span_max)
            c0 = random.randint(1, w - span + 1)
            row = 0 if side == "T" else h + 1
            canvas[row, c0 : c0 + span] = True
        else:
            span_max = max(1, min(h, max(1, h // 2)))
            span = random.randint(1, span_max)
            r0 = random.randint(1, h - span + 1)
            col = 0 if side == "L" else w + 1
            canvas[r0 : r0 + span, col] = True
    return _tight_bool(canvas)


def _render_tree_patch(tree: Tree) -> np.ndarray:
    kind, children = tree
    if kind == "B":
        return _render_blob_patch()

    child_patches = [_render_tree_patch(child) for child in children]
    child_sizes = [tuple(map(int, patch.shape)) for patch in child_patches]
    packed = _pack_rectangles(
        child_sizes,
        gap=1,
        min_h=1,
        min_w=1,
        max_h=10,
        max_w=10,
    )
    if packed is None:
        raise ValueError("children did not fit")
    content_h, content_w, positions = packed

    patch = np.zeros((content_h + 4, content_w + 4), dtype=bool)
    patch[0, :] = True
    patch[-1, :] = True
    patch[:, 0] = True
    patch[:, -1] = True
    for child_patch, (r, c) in zip(child_patches, positions):
        h, w = child_patch.shape
        patch[r + 2 : r + 2 + h, c + 2 : c + 2 + w] |= child_patch
    patch = _add_frame_protrusions(patch)
    return _random_transform(patch)


def _paint_patch(mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    return np.where(mask, int(fg), int(bg)).astype(int)


def _sample_base_and_outlier() -> tuple[Tree, Tree]:
    if random.random() < 0.6:
        return random.choice(CHALLENGE_PAIRS)
    base = random.choice(TREE_LIBRARY)
    outlier_choices = [tree for tree in TREE_LIBRARY if _tree_signature_from_template(tree) != _tree_signature_from_template(base)]
    return base, random.choice(outlier_choices)


def _build_structure_patches() -> tuple[list[np.ndarray], int, np.ndarray] | None:
    base_tree, outlier_tree = _sample_base_and_outlier()
    structure_count = random.randint(3, 4)
    outlier_index = random.randrange(structure_count)
    trees = [base_tree] * structure_count
    trees[outlier_index] = outlier_tree

    sigs = [_tree_signature_from_template(tree) for tree in trees]
    sig_counts = Counter(sigs)
    if len(sig_counts) != 2 or sorted(sig_counts.values()) != [1, structure_count - 1]:
        return None

    patches: list[np.ndarray] = []
    for tree in trees:
        patch = None
        for _ in range(MAX_PATCH_RENDER_ATTEMPTS):
            try:
                candidate = _render_tree_patch(tree)
            except ValueError as err:
                if str(err) != "children did not fit":
                    raise
                continue
            if max(candidate.shape) > MAX_STRUCTURE_SIDE:
                continue
            patch = candidate
            break
        if patch is None:
            return None
        patches.append(patch)

    hashes = {patch.shape + tuple(patch.ravel().tolist()) for patch in patches}
    if len(hashes) != len(patches):
        return None

    outlier_patch = patches[outlier_index]
    return patches, outlier_index, outlier_patch


def _place_structures(patches: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]]] | None:
    sizes = [tuple(map(int, patch.shape)) for patch in patches]
    packed = _pack_rectangles(
        sizes,
        gap=2,
        min_h=max(max(h for h, _w in sizes), 16),
        min_w=max(max(w for _h, w in sizes), 16),
        max_h=MAX_GRID_SIDE,
        max_w=MAX_GRID_SIDE,
        tries_per_canvas=60,
    )
    if packed is None:
        return None
    base_h, base_w, positions = packed
    total_fg = sum(int(patch.sum()) for patch in patches)

    for _ in range(30):
        extra_h = random.randint(0, MAX_GRID_SIDE - base_h)
        extra_w = random.randint(0, MAX_GRID_SIDE - base_w)
        grid_h = base_h + extra_h
        grid_w = base_w + extra_w
        if grid_h * grid_w <= total_fg * 2:
            continue
        shift_r = random.randint(0, extra_h)
        shift_c = random.randint(0, extra_w)
        shifted = [(r + shift_r, c + shift_c) for r, c in positions]
        return grid_h, grid_w, shifted
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    try:
        built = _build_structure_patches()
    except ValueError as err:
        if str(err) != "children did not fit":
            raise
        return None
    if built is None:
        return None
    patches, outlier_index, outlier_patch = built
    placed = _place_structures(patches)
    if placed is None:
        return None
    grid_h, grid_w, positions = placed

    colors = list(range(10))
    bg = random.choice(colors)
    fg = random.choice([color for color in colors if color != bg])
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    for patch, (r0, c0) in zip(patches, positions):
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w][patch] = fg

    expected = _paint_patch(outlier_patch, bg, fg)
    if max(grid.shape) > MAX_GRID_SIDE or max(expected.shape) > MAX_GRID_SIDE:
        return None
    if _get_bg(grid) != bg:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid topology-outlier ARC input")
