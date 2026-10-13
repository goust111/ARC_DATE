from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _part_from_coords(coords: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    local = coords - np.array([[r0, c0]], dtype=int)
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "local": local,
        "shape": (r1 - r0 + 1, c1 - c0 + 1),
        "center": (float(coords[:, 0].mean()), float(coords[:, 1].mean())),
    }


def _placement_masks(local: np.ndarray, shape: tuple[int, int], rect_h: int, rect_w: int) -> list[int]:
    part_h, part_w = shape
    base = [int(r) * rect_w + int(c) for r, c in local]
    masks: list[int] = []
    for dr in range(rect_h - part_h + 1):
        row_shift = dr * rect_w
        for dc in range(rect_w - part_w + 1):
            mask = 0
            shift = row_shift + dc
            for idx in base:
                mask |= 1 << (idx + shift)
            masks.append(mask)
    return masks


def _candidate_rectangle_shapes(part_a: dict[str, object], part_b: dict[str, object]) -> list[tuple[int, int]]:
    local_a = np.asarray(part_a["local"], dtype=int)
    local_b = np.asarray(part_b["local"], dtype=int)
    shape_a = tuple(map(int, part_a["shape"]))
    shape_b = tuple(map(int, part_b["shape"]))
    area = int(local_a.shape[0] + local_b.shape[0])
    max_h = max(shape_a[0], shape_b[0])
    max_w = max(shape_a[1], shape_b[1])
    shapes: list[tuple[int, int]] = []

    for rect_h in range(max_h, area + 1):
        if area % rect_h != 0:
            continue
        rect_w = area // rect_h
        if rect_w < max_w:
            continue

        masks_a = _placement_masks(local_a, shape_a, rect_h, rect_w)
        masks_b = _placement_masks(local_b, shape_b, rect_h, rect_w)
        if any((mask_a & mask_b) == 0 for mask_a in masks_a for mask_b in masks_b):
            shapes.append((rect_h, rect_w))

    return shapes


def _find_hint_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    counts = {int(color): int(np.count_nonzero(grid == color)) for color in np.unique(grid)}
    candidates: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            sub = grid[r : r + 2, c : c + 2]
            a = int(sub[0, 0])
            b = int(sub[0, 1])
            c1 = int(sub[1, 0])
            d = int(sub[1, 1])
            if a == bg or b == bg:
                continue
            if a == d and b == c1 and a != b and counts.get(a) == 2 and counts.get(b) == 2:
                candidates.append((r, c))

    assert len(candidates) == 1, "expected one 2x2 diagonal hint area"
    r0, c0 = candidates[0]
    hint_mask = np.zeros(grid.shape, dtype=bool)
    hint_mask[r0 : r0 + 2, c0 : c0 + 2] = True
    return hint_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each same-color fragment pair into the unique solid rectangle that they tile.

    The 2x2 diagonal hint block is preserved unchanged. Outside that block, every
    non-background color appears in exactly two connected components whose shapes are
    complementary: after translating one component, the pair can tile one axis-aligned
    rectangle. The solved rectangle is painted at the top-left corner of the two
    fragments' bbox union, and the original fragment pixels are cleared.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _majority_color(g)
    hint_mask = _find_hint_mask(g, bg)

    work = g.copy()
    work[hint_mask] = bg

    pairs: list[tuple[int, int, int, int, int, list[np.ndarray]]] = []
    for color in map(int, np.unique(work)):
        if color == bg:
            continue
        comps = _components(work == color)
        assert len(comps) == 2, "each non-hint color must split into exactly two fragments"
        parts = [_part_from_coords(comp) for comp in comps]
        for part in parts:
            part_h, part_w = map(int, part["shape"])
            assert len(part["coords"]) < part_h * part_w, "fragments must be non-rectangular"
        parts.sort(key=lambda part: tuple(part["center"]))

        shapes = _candidate_rectangle_shapes(parts[0], parts[1])
        assert len(shapes) == 1, "each fragment pair must tile a unique rectangle"
        rect_h, rect_w = shapes[0]
        top = min(int(parts[0]["bbox"][0]), int(parts[1]["bbox"][0]))
        left = min(int(parts[0]["bbox"][1]), int(parts[1]["bbox"][1]))
        assert top + rect_h <= g.shape[0] and left + rect_w <= g.shape[1]
        pairs.append((top, left, rect_h, rect_w, color, [parts[0]["coords"], parts[1]["coords"]]))

    out = g.copy()
    for _top, _left, _rect_h, _rect_w, _color, comp_coords in pairs:
        for coords in comp_coords:
            out[coords[:, 0], coords[:, 1]] = bg

    occupied = hint_mask.copy()
    for top, left, rect_h, rect_w, color, _comp_coords in sorted(pairs):
        assert not occupied[top : top + rect_h, left : left + rect_w].any(), "merged rectangles must not overlap"
        out[top : top + rect_h, left : left + rect_w] = color
        occupied[top : top + rect_h, left : left + rect_w] = True

    out[hint_mask] = g[hint_mask]
    return out


# --------------- generator ---------------

BG = 0
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 150
PAIR_COUNT_RANGE = (2, 4)
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (20, 30)
RECT_H_RANGE = (3, 7)
RECT_W_RANGE = (4, 7)
BOX_PAD = 1


def _box_free(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = BOX_PAD) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _reserve_box(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = BOX_PAD) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _stamp_shape(grid: np.ndarray, local: np.ndarray, top: int, left: int, color: int) -> None:
    rr = local[:, 0] + int(top)
    cc = local[:, 1] + int(left)
    grid[rr, cc] = int(color)


def _sample_pair_spec() -> dict[str, object] | None:
    for _ in range(120):
        rect_h = random.randint(*RECT_H_RANGE)
        rect_w = random.randint(*RECT_W_RANGE)
        thresholds = [random.randint(1, rect_w - 1) for _ in range(rect_h)]
        if len(set(thresholds)) < 2:
            continue
        if min(thresholds) > rect_w - 2 or max(thresholds) < 2:
            continue
        if random.random() < 0.5:
            thresholds.reverse()

        mask_a = np.zeros((rect_h, rect_w), dtype=bool)
        for r, width in enumerate(thresholds):
            mask_a[r, :width] = True
        mask_b = ~mask_a

        area_a = int(mask_a.sum())
        area_b = int(mask_b.sum())
        if min(area_a, area_b) < 4:
            continue

        part_a = _part_from_coords(np.argwhere(mask_a))
        part_b = _part_from_coords(np.argwhere(mask_b))
        shapes = _candidate_rectangle_shapes(part_a, part_b)
        if shapes != [(rect_h, rect_w)]:
            continue

        return {
            "rect_shape": (rect_h, rect_w),
            "a_local": np.asarray(part_a["local"], dtype=int),
            "a_shape": tuple(map(int, part_a["shape"])),
            "b_local": np.asarray(part_b["local"], dtype=int),
            "b_shape": tuple(map(int, part_b["shape"])),
        }

    return None


def _place_hint(grid: np.ndarray, occupied: np.ndarray, colors: tuple[int, int]) -> bool:
    h, w = grid.shape
    side = random.choice(("top", "bottom", "left", "right"))
    for _ in range(50):
        if side == "top":
            top = 0
            left = random.randint(1, w - 3)
        elif side == "bottom":
            top = h - 2
            left = random.randint(1, w - 3)
        elif side == "left":
            top = random.randint(1, h - 3)
            left = 0
        else:
            top = random.randint(1, h - 3)
            left = w - 2

        if not _box_free(occupied, top, left, 2, 2):
            continue
        a, b = map(int, colors)
        grid[top : top + 2, left : left + 2] = np.array([[a, b], [b, a]], dtype=int)
        _reserve_box(occupied, top, left, 2, 2)
        return True

    return False


def _place_pair(
    inp: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    color: int,
    spec: dict[str, object],
    branch: str,
) -> bool:
    grid_h, grid_w = inp.shape
    rect_h, rect_w = map(int, spec["rect_shape"])
    b_h, b_w = map(int, spec["b_shape"])
    a_local = np.asarray(spec["a_local"], dtype=int)
    b_local = np.asarray(spec["b_local"], dtype=int)

    for _ in range(PAIR_ATTEMPTS):
        top = random.randint(0, grid_h - rect_h)
        left = random.randint(0, grid_w - rect_w)
        if not _box_free(occupied, top, left, rect_h, rect_w):
            continue

        if branch == "right":
            b_top = top
            b_left_min = left + rect_w + 1
            b_left_max = grid_w - b_w
            if b_left_min > b_left_max:
                continue
            b_left = random.randint(b_left_min, b_left_max)
        else:
            b_top_min = top + rect_h + 1
            b_top_max = grid_h - b_h
            if b_top_min > b_top_max:
                continue
            b_top = random.randint(b_top_min, b_top_max)
            b_left_min = left
            b_left_max = grid_w - b_w
            if b_left_min > b_left_max:
                continue
            b_left = random.randint(b_left_min, b_left_max)

        if not _box_free(occupied, b_top, b_left, b_h, b_w):
            continue

        _stamp_shape(inp, a_local, top, left, color)
        _stamp_shape(inp, b_local, b_top, b_left, color)
        expected[top : top + rect_h, left : left + rect_w] = color
        _reserve_box(occupied, top, left, rect_h, rect_w)
        _reserve_box(occupied, b_top, b_left, b_h, b_w)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)

    available = list(range(1, 10))
    if pair_count + 2 > len(available):
        return None
    chosen = random.sample(available, pair_count + 2)
    hint_colors = (chosen[0], chosen[1])
    pair_colors = chosen[2:]

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    if not _place_hint(inp, occupied, hint_colors):
        return None
    expected[:, :] = inp

    branches = [random.choice(("right", "down")) for _ in range(pair_count)]
    random.shuffle(branches)

    for color, branch in zip(pair_colors, branches):
        spec = _sample_pair_spec()
        if spec is None:
            return None
        if not _place_pair(inp, expected, occupied, color, spec, branch):
            return None

    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
