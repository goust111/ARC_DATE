from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
BLUE = 1
RED = 2

SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "L3_LEFT": ((0, 0), (1, 0), (1, 1)),
    "L3_RIGHT": ((0, 1), (1, 0), (1, 1)),
    "T4": ((0, 1), (1, 0), (1, 1), (1, 2)),
    "Z4": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "O4": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "U5": ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
}

SHAPE_SET_OPTIONS: dict[int, tuple[tuple[str, ...], ...]] = {
    2: (
        ("L3_LEFT", "L3_RIGHT"),
        ("T4", "Z4"),
    ),
    3: (
        ("L3_LEFT", "L3_RIGHT", "O4"),
        ("T4", "Z4", "O4"),
        ("L3_LEFT", "L3_RIGHT", "T4"),
        ("T4", "Z4", "U5"),
    ),
}

COUNT_OPTIONS: dict[int, tuple[tuple[int, ...], ...]] = {
    2: (
        (3, 5),
        (4, 5),
    ),
    3: (
        (3, 3, 4),
        (3, 4, 4),
        (3, 4, 5),
        (5, 3, 4),
    ),
}

BORDER = 1
GRID_HEIGHT_RANGE = (16, 22)
GRID_WIDTH_PAD_RANGE = (4, 10)
GENERATION_RETRIES = 500
SEQUENCE_RETRIES = 300


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(rel)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _objects_from_grid(grid: np.ndarray) -> list[dict]:
    comps = _find_components(grid == GREY)
    assert 8 <= len(comps) <= 12, "expected 8-12 grey objects"

    objects = []
    for coords in comps:
        bbox = _bbox(coords)
        key = _shape_key(coords)
        r0, c0, r1, c1 = bbox
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "shape": key,
                "area": int(coords.shape[0]),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )
    return objects


def _group_objects(objects: list[dict], mode: str) -> dict[object, list[dict]]:
    groups: dict[object, list[dict]] = defaultdict(list)
    for obj in objects:
        if mode == "shape":
            key: object = obj["shape"]
        elif mode == "bbox":
            key = (obj["height"], obj["width"])
        elif mode == "area":
            key = obj["area"]
        elif mode == "global":
            key = "all"
        else:
            raise ValueError(f"unknown grouping mode: {mode}")
        groups[key].append(obj)
    return groups


def _recolor(
    grid: np.ndarray,
    objects: list[dict],
    *,
    group_mode: str,
    sort_mode: str,
    anchor_side: str,
) -> np.ndarray:
    out = grid.copy()
    groups = _group_objects(objects, group_mode)

    for members in groups.values():
        if sort_mode == "col":
            ordered = sorted(members, key=lambda obj: obj["bbox"][1])
            cols = [obj["bbox"][1] for obj in ordered]
            assert len(set(cols)) == len(cols), "group columns must be unique"
        elif sort_mode == "row":
            ordered = sorted(members, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
        else:
            raise ValueError(f"unknown sort mode: {sort_mode}")

        walk = list(reversed(ordered)) if anchor_side == "right" else ordered
        for rank, obj in enumerate(walk):
            color = RED if rank % 2 == 0 else BLUE
            coords = obj["coords"]
            out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each same-shape grey group with right-anchored alternating colors.

    Input structure:
    - A black background contains 8-12 disjoint grey 4-connected objects.
    - Every object has 3-6 cells, and the normalized object masks belong to 2-3 repeated shape types.
    - Objects with the same shape appear at different positions without rotation or reflection.

    Rule:
    - Extract every grey 4-connected object and translate-normalize its cell coordinates by its own bbox top-left corner.
    - Group objects whose normalized masks are exactly equal.
    - Inside each shape group, sort objects by the bbox top-left column from left to right.
    - Recolor the rightmost, third-from-right, fifth-from-right, ... objects red and the alternating ones blue.
    - Leave the background unchanged.

    Key insight:
    - The alternation is not global: each normalized shape creates its own horizontal ranking space, so shape grouping is required before parity coloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid exceeds ARC size limit"
    colors = {int(v) for v in np.unique(g)}
    assert colors <= {BG, GREY}, "expected only black background and grey objects"

    objects = _objects_from_grid(g)
    shape_groups = _group_objects(objects, "shape")
    assert 2 <= len(shape_groups) <= 3, "expected 2-3 repeated shape groups"
    for members in shape_groups.values():
        assert 3 <= len(members) <= 5, "each shape should appear 3-5 times"

    return _recolor(g, objects, group_mode="shape", sort_mode="col", anchor_side="right")


# --------------- generator ---------------

_RNG = np.random.default_rng()


def _shape_dims(shape_name: str) -> tuple[int, int]:
    coords = SHAPES[shape_name]
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    return h, w


def _shape_key_from_name(shape_name: str) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(SHAPES[shape_name]))


def _sample_label_sequence(counts: tuple[int, ...]) -> list[int] | None:
    expanded = []
    for label, count in enumerate(counts):
        expanded.extend([label] * count)

    for _ in range(SEQUENCE_RETRIES):
        seq = list(_RNG.permutation(expanded))
        if any(a == b for a, b in zip(seq, seq[1:])):
            continue

        runs = {label: 0 for label in range(len(counts))}
        prev = None
        for label in seq:
            if label != prev:
                runs[int(label)] += 1
                prev = int(label)
        if any(run_count < 2 for run_count in runs.values()):
            continue

        if not any(seq[i] == seq[i + 2] and seq[i] != seq[i + 1] for i in range(len(seq) - 2)):
            continue

        return [int(x) for x in seq]
    return None


def _abs_cells(shape_name: str, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in SHAPES[shape_name]]


def _fits(
    occupied: np.ndarray,
    shape_name: str,
    top: int,
    left: int,
) -> bool:
    h, w = occupied.shape
    for r, c in _abs_cells(shape_name, top, left):
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _assign_rows(objects: list[dict], h: int, w: int) -> list[int] | None:
    occupied = np.zeros((h, w), dtype=bool)
    tops: list[int | None] = [None] * len(objects)

    row_options = {
        idx: list(range(BORDER, h - objects[idx]["height"] - BORDER + 1))
        for idx in range(len(objects))
    }

    def place(idx: int, top: int) -> list[tuple[int, int]]:
        cells = _abs_cells(objects[idx]["shape_name"], top, objects[idx]["left"])
        for r, c in cells:
            occupied[r, c] = True
        tops[idx] = top
        return cells

    def unplace(idx: int, cells: list[tuple[int, int]]) -> None:
        for r, c in cells:
            occupied[r, c] = False
        tops[idx] = None

    def backtrack() -> bool:
        remaining = [idx for idx, top in enumerate(tops) if top is None]
        if not remaining:
            return True

        best_idx = None
        best_rows = None
        for idx in remaining:
            candidates = [top for top in row_options[idx] if _fits(occupied, objects[idx]["shape_name"], top, objects[idx]["left"])]
            if not candidates:
                return False
            _RNG.shuffle(candidates)
            if best_rows is None or len(candidates) < len(best_rows):
                best_idx = idx
                best_rows = candidates

        assert best_idx is not None and best_rows is not None
        for top in best_rows:
            cells = place(best_idx, int(top))
            if backtrack():
                return True
            unplace(best_idx, cells)
        return False

    return [int(x) for x in tops] if backtrack() else None


def _render_scene(objects: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = GREY
    return grid


def _rows_shortcut_differs(expected: np.ndarray, grid: np.ndarray, objects: list[dict]) -> bool:
    row_sorted = _recolor(grid, objects, group_mode="shape", sort_mode="row", anchor_side="right")
    return not np.array_equal(expected, row_sorted)


def _group_sizes(labels: list[int]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    return dict(counts)


def _gen_attempt() -> np.ndarray | None:
    n_groups = 3 if _RNG.random() < 0.6 else 2
    shape_names = list(random.choice(SHAPE_SET_OPTIONS[n_groups]))
    counts = tuple(int(x) for x in random.choice(COUNT_OPTIONS[n_groups]))
    sequence = _sample_label_sequence(counts)
    if sequence is None:
        return None

    n_objects = len(sequence)
    if not 8 <= n_objects <= 12:
        return None

    max_width = max(_shape_dims(name)[1] for name in shape_names)
    min_width = n_objects + max_width + 1
    width_lo = max(15, min_width)
    width_hi = min(26, n_objects + GRID_WIDTH_PAD_RANGE[1])
    if width_lo > width_hi:
        return None
    width = int(_RNG.integers(width_lo, width_hi + 1))
    height = int(_RNG.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))

    available_cols = np.arange(BORDER, width - max_width - BORDER + 1)
    if available_cols.size < n_objects:
        return None
    left_cols = sorted(int(x) for x in _RNG.choice(available_cols, size=n_objects, replace=False))

    objects = []
    for idx, (label, left) in enumerate(zip(sequence, left_cols)):
        shape_name = shape_names[label]
        shape_h, shape_w = _shape_dims(shape_name)
        objects.append(
            {
                "index": idx,
                "label": int(label),
                "shape_name": shape_name,
                "shape": _shape_key_from_name(shape_name),
                "height": shape_h,
                "width": shape_w,
                "left": int(left),
            }
        )

    tops = _assign_rows(objects, height, width)
    if tops is None:
        return None

    for obj, top in zip(objects, tops):
        obj["top"] = int(top)
        obj["cells"] = _abs_cells(obj["shape_name"], int(top), obj["left"])
        obj["bbox"] = (
            int(top),
            obj["left"],
            int(top + obj["height"] - 1),
            int(obj["left"] + obj["width"] - 1),
        )
        obj["coords"] = np.array(obj["cells"], dtype=int)
        obj["area"] = len(obj["cells"])

    grid = _render_scene(objects, height, width)
    expected = _recolor(grid, objects, group_mode="shape", sort_mode="col", anchor_side="right")

    # Reject scenes that admit common wrong shortcuts.
    if np.array_equal(expected, _recolor(grid, objects, group_mode="global", sort_mode="col", anchor_side="right")):
        return None
    if np.array_equal(expected, _recolor(grid, objects, group_mode="shape", sort_mode="col", anchor_side="left")):
        return None
    if np.array_equal(expected, _recolor(grid, objects, group_mode="bbox", sort_mode="col", anchor_side="right")):
        return None
    if np.array_equal(expected, _recolor(grid, objects, group_mode="area", sort_mode="col", anchor_side="right")):
        return None
    if not _rows_shortcut_differs(expected, grid, objects):
        return None

    # Structural checks that enforce the intended training signal.
    label_counts = _group_sizes(sequence)
    if not any(count % 2 == 0 for count in label_counts.values()):
        return None
    if not any(count % 2 == 1 for count in label_counts.values()):
        return None
    if not any(count >= 4 for count in label_counts.values()):
        return None

    by_label = defaultdict(list)
    for obj in objects:
        by_label[obj["label"]].append(obj)
    if any(len({item["left"] for item in members}) != len(members) for members in by_label.values()):
        return None

    # Require at least one group whose row ordering disagrees with column ordering.
    row_vs_col_diff = False
    for members in by_label.values():
        if len(members) < 3:
            continue
        cols = [item["left"] for item in sorted(members, key=lambda item: item["left"])]
        rows = [item["top"] for item in sorted(members, key=lambda item: item["left"])]
        if rows != sorted(rows) and rows != sorted(rows, reverse=True):
            row_vs_col_diff = True
            break
    if not row_vs_col_diff:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
