from __future__ import annotations

import itertools
import random

import numpy as np


INPUT_BG = 5
OUTPUT_BG = 0
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ROW_RANGE = (11, 16)
GEN_COL_RANGE = (13, 18)
GEN_OBJECT_COUNT = (3, 4)
GEN_ATTEMPTS = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        count += 1
        stack = [(sr, sc)]
        seen[sr, sc] = True
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _has_lr_symmetry(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[:, ::-1]))


def _has_ud_symmetry(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[::-1, :]))


def _complete_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "horizontal":
        return mask | mask[::-1, :]
    if axis == "vertical":
        return mask | mask[:, ::-1]
    raise ValueError(axis)


def _solve(grid: np.ndarray, mode: str = "detect") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = np.full_like(g, bg)
    for color in _colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert len(coords) > 0
        r0, c0, r1, c1 = _bbox_coords(coords)
        local = g[r0 : r1 + 1, c0 : c1 + 1] == color
        lr = _has_lr_symmetry(local)
        ud = _has_ud_symmetry(local)

        if mode == "detect":
            assert lr ^ ud, f"each mask must have exactly one symmetry: color={color}"
            axis = "horizontal" if lr else "vertical"
            completed = _complete_mask(local, axis)
        elif mode == "horizontal":
            completed = _complete_mask(local, "horizontal")
        elif mode == "vertical":
            completed = _complete_mask(local, "vertical")
        elif mode == "none":
            completed = local.copy()
        else:
            raise ValueError(mode)

        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[completed] = color

    fg_bbox = _bbox(out != bg)
    assert fg_bbox is not None, "expected foreground masks"
    cropped = _crop_bbox(out, fg_bbox)
    cropped[cropped == bg] = OUTPUT_BG
    return cropped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each color mask's existing symmetry type and complete it along the other axis before cropping.

    Input structure:
    - The grid has a gray background and 3 or 4 different foreground colors.
    - Each foreground color forms one small 4-connected monochrome mask.
    - Every mask is isolated from the others and occupies its own tight bounding box.

    Rule:
    - For each non-background color independently, compute its tight bounding box and inspect the mask only inside that box.
    - If the mask already has left-right symmetry, mirror its pixels across the horizontal midline inside the same box.
    - If the mask already has up-down symmetry, mirror its pixels across the vertical midline inside the same box.
    - After all colors are completed, crop to the tight bounding box of all foreground pixels and replace the gray background inside that crop with black.

    Key insight:
    - The visible symmetry is not the final target; it is the clue that tells you which perpendicular completion to perform.
    """
    return _solve(grid, mode="detect")


def _palindromic_rows(width: int) -> list[np.ndarray]:
    half = (width + 1) // 2
    rows: list[np.ndarray] = []
    for bits in range(1, 1 << half):
        left = np.array([(bits >> i) & 1 for i in range(half)], dtype=bool)
        if width % 2 == 0:
            row = np.concatenate([left, left[::-1]])
        else:
            row = np.concatenate([left, left[-2::-1]])
        rows.append(row)
    return rows


def _touches_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _valid_single_axis_mask(mask: np.ndarray, symmetry: str) -> bool:
    area = int(mask.sum())
    if not (4 <= area <= 8):
        return False
    if not _touches_bbox(mask):
        return False
    if _component_count(mask) != 1:
        return False

    lr = _has_lr_symmetry(mask)
    ud = _has_ud_symmetry(mask)
    if symmetry == "lr":
        if not (lr and not ud):
            return False
        completed = _complete_mask(mask, "horizontal")
    else:
        if not (ud and not lr):
            return False
        completed = _complete_mask(mask, "vertical")

    if np.array_equal(mask, completed):
        return False
    if _component_count(completed) != 1:
        return False
    if completed.all():
        return False
    return bool(_has_lr_symmetry(completed) and _has_ud_symmetry(completed))


def _build_lr_catalog() -> list[np.ndarray]:
    catalog: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for height in range(2, 6):
        for width in range(3, 6):
            rows = _palindromic_rows(width)
            for combo in itertools.product(range(len(rows)), repeat=height):
                mask = np.stack([rows[i] for i in combo], axis=0)
                if not _valid_single_axis_mask(mask, "lr"):
                    continue
                key = (mask.shape, mask.tobytes())
                if key not in seen:
                    seen.add(key)
                    catalog.append(mask.copy())
    assert catalog, "expected non-empty left-right-symmetric catalog"
    return catalog


LR_CATALOG = _build_lr_catalog()
UD_CATALOG = []
_ud_seen: set[tuple[tuple[int, int], bytes]] = set()
for _mask in LR_CATALOG:
    _t = _mask.T.copy()
    if _valid_single_axis_mask(_t, "ud"):
        _key = (_t.shape, _t.tobytes())
        if _key not in _ud_seen:
            _ud_seen.add(_key)
            UD_CATALOG.append(_t)
assert UD_CATALOG, "expected non-empty up-down-symmetric catalog"


def _make_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, INPUT_BG, dtype=int)
    patch[mask] = color
    return patch


def _bbox_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (
        a[2] + 1 < b[0]
        or b[2] + 1 < a[0]
        or a[3] + 1 < b[1]
        or b[3] + 1 < a[1]
    )


def _place_patches(shape: tuple[int, int], patches: list[np.ndarray]) -> np.ndarray | None:
    h, w = shape
    grid = np.full(shape, INPUT_BG, dtype=int)
    reserved: list[tuple[int, int, int, int]] = []

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_bbox_too_close(bbox, prev) for prev in reserved):
                continue
            view = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any(view != INPUT_BG):
                continue
            view[patch != INPUT_BG] = patch[patch != INPUT_BG]
            reserved.append(bbox)
            placed = True
            break

        if not placed:
            return None

    return grid


def _object_summary(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], str, np.ndarray]]:
    info = []
    for color in _colors_present(grid, exclude={INPUT_BG}):
        mask = grid == color
        assert _component_count(mask) == 1, f"color {color} must form one component"
        bbox = _bbox(mask)
        assert bbox is not None
        local = _crop_bbox(mask, bbox)
        lr = _has_lr_symmetry(local)
        ud = _has_ud_symmetry(local)
        assert lr ^ ud, f"color {color} must have exactly one symmetry"
        info.append((color, bbox, "lr" if lr else "ud", local))
    return info


def _validate_scene(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if _get_bg(grid) != INPUT_BG:
        return False

    colors = _colors_present(grid, exclude={INPUT_BG})
    if len(colors) not in {3, 4}:
        return False

    try:
        objects = _object_summary(grid)
    except AssertionError:
        return False

    types = {sym for _color, _bbox0, sym, _local in objects}
    if types != {"lr", "ud"}:
        return False

    bboxes = [bbox for _color, bbox, _sym, _local in objects]
    for i, bbox_a in enumerate(bboxes):
        for bbox_b in bboxes[i + 1 :]:
            if _bbox_too_close(bbox_a, bbox_b):
                return False

    for _color, _bbox0, sym, local in objects:
        completed = _complete_mask(local, "horizontal" if sym == "lr" else "vertical")
        if completed.all():
            return False
        if np.array_equal(local, completed):
            return False
        if not (_has_lr_symmetry(completed) and _has_ud_symmetry(completed)):
            return False

    try:
        actual = main(grid)
        wrong_horizontal = _solve(grid, mode="horizontal")
        wrong_vertical = _solve(grid, mode="vertical")
        unchanged = _solve(grid, mode="none")
    except AssertionError:
        return False

    if actual.shape != main(grid).shape:
        return False
    if np.array_equal(actual, wrong_horizontal):
        return False
    if np.array_equal(actual, wrong_vertical):
        return False
    if np.array_equal(actual, unchanged):
        return False
    if not np.any(actual == OUTPUT_BG):
        return False
    return True


def _random_types(count: int) -> list[str]:
    types = ["lr", "ud"]
    while len(types) < count:
        types.append(random.choice(["lr", "ud"]))
    random.shuffle(types)
    return types


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GEN_ROW_RANGE)
    cols = random.randint(*GEN_COL_RANGE)
    count = random.randint(*GEN_OBJECT_COUNT)
    colors = random.sample(list(OBJECT_COLORS), count)
    types = _random_types(count)

    patches: list[np.ndarray] = []
    for color, symmetry in zip(colors, types):
        catalog = LR_CATALOG if symmetry == "lr" else UD_CATALOG
        mask = random.choice(catalog)
        patches.append(_make_patch(mask, color))

    grid = _place_patches((rows, cols), patches)
    if grid is None:
        return None
    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
