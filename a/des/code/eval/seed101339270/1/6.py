from __future__ import annotations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _square_infos(grid: np.ndarray, bg: int) -> list[dict]:
    infos: list[dict] = []
    for coords in _components(grid != bg):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        size_r = int(r1 - r0 + 1)
        size_c = int(c1 - c0 + 1)
        if size_r != size_c:
            continue
        if int(coords.shape[0]) != size_r * size_c:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        infos.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "size": int(size_r),
                "patch": patch,
            }
        )
    return infos


def _border_values(patch: np.ndarray) -> np.ndarray:
    if patch.shape[0] == 1:
        return patch[0, :].copy()
    top = patch[0, :]
    bottom = patch[-1, :]
    left = patch[1:-1, 0]
    right = patch[1:-1, -1]
    return np.concatenate([top, bottom, left, right])


def _project_axis(mark0: int, size: int, master0: int, master1: int) -> int:
    mark1 = mark0 + size - 1
    if mark1 < master0:
        return 2 * master0 - mark0 - size
    if mark0 > master1:
        return 2 * (master1 + 1) - mark0 - size
    return mark0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique largest filled square as the master, then swap every smaller
    same-colored square with the same-sized region projected into the master.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    squares = _square_infos(g, bg)
    assert squares, "expected at least one filled square object"

    max_size = max(info["size"] for info in squares)
    masters = [info for info in squares if info["size"] == max_size]
    assert len(masters) == 1, "expected a unique largest square"
    master = masters[0]
    mr0, mc0, mr1, mc1 = master["bbox"]
    master_patch = master["patch"]

    border = _border_values(master_patch)
    colors, counts = np.unique(border, return_counts=True)
    base_color = int(colors[int(np.argmax(counts))])
    assert np.all(border == base_color), "master border must reveal the shared square color"

    marks = []
    for info in squares:
        if info is master or info["size"] >= master["size"]:
            continue
        vals = np.unique(info["patch"])
        if len(vals) == 1 and int(vals[0]) == base_color:
            marks.append(info)
    assert marks, "expected at least one smaller same-colored square"

    original = g.copy()
    out = g.copy()
    claimed = np.zeros(master_patch.shape, dtype=bool)
    transfers: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []

    for info in marks:
        size = info["size"]
        sr0, sc0, sr1, sc1 = info["bbox"]
        tr0 = _project_axis(sr0, size, mr0, mr1)
        tc0 = _project_axis(sc0, size, mc0, mc1)
        tr1 = tr0 + size - 1
        tc1 = tc0 + size - 1

        assert mr0 <= tr0 <= tr1 <= mr1
        assert mc0 <= tc0 <= tc1 <= mc1
        rel = claimed[tr0 - mr0 : tr1 - mr0 + 1, tc0 - mc0 : tc1 - mc0 + 1]
        assert not np.any(rel), "projected windows must not overlap"
        rel[:] = True

        source = original[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = source
        transfers.append(((tr0, tc0, tr1, tc1), (sr0, sc0, sr1, sc1)))

    for (tr0, tc0, tr1, tc1), _mark_box in transfers:
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = bg

    return out


_GRID_MIN = 18
_GRID_MAX = 30
_MASTER_MIN = 6
_MASTER_MAX = 10
_MARK_COUNT_RANGE = (1, 4)
_DECORATION_SHAPES = ("rect", "hline", "vline", "plus", "diag", "dots", "checker")
_DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
_GEN_ATTEMPTS = 400


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _sample_colors() -> tuple[int, int, list[int]]:
    palette = np.random.permutation(10)
    bg = int(palette[0])
    base = int(palette[1])
    deco_count = _randint(1, 3)
    decos = [int(x) for x in palette[2 : 2 + deco_count]]
    return bg, base, decos


def _paint_rect(patch: np.ndarray, color: int) -> None:
    h, w = patch.shape
    rh = _randint(1, max(1, min(4, h - 2)))
    rw = _randint(1, max(1, min(4, w - 2)))
    r0 = _randint(1, h - rh - 1)
    c0 = _randint(1, w - rw - 1)
    patch[r0 : r0 + rh, c0 : c0 + rw] = color


def _paint_hline(patch: np.ndarray, color: int) -> None:
    h, w = patch.shape
    r = _randint(1, h - 2)
    c0 = _randint(1, w - 2)
    c1 = _randint(c0, w - 2)
    patch[r, c0 : c1 + 1] = color


def _paint_vline(patch: np.ndarray, color: int) -> None:
    h, w = patch.shape
    c = _randint(1, w - 2)
    r0 = _randint(1, h - 2)
    r1 = _randint(r0, h - 2)
    patch[r0 : r1 + 1, c] = color


def _paint_plus(patch: np.ndarray, color: int) -> None:
    h, w = patch.shape
    r = _randint(1, h - 2)
    c = _randint(1, w - 2)
    patch[r, max(1, c - 1) : min(w - 1, c + 2)] = color
    patch[max(1, r - 1) : min(h - 1, r + 2), c] = color


def _paint_diag(patch: np.ndarray, color: int) -> None:
    h, w = patch.shape
    length = _randint(2, max(2, min(h - 2, w - 2)))
    forward = bool(_randint(0, 1))
    if forward:
        r0 = _randint(1, h - length - 1)
        c0 = _randint(1, w - length - 1)
        for i in range(length):
            patch[r0 + i, c0 + i] = color
    else:
        r0 = _randint(1, h - length - 1)
        c0 = _randint(length, w - 2)
        for i in range(length):
            patch[r0 + i, c0 - i] = color


def _paint_dots(patch: np.ndarray, color: int) -> None:
    h, w = patch.shape
    count = _randint(2, min(6, (h - 2) * (w - 2)))
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    order = np.random.permutation(len(cells))[:count]
    for idx in order:
        r, c = cells[int(idx)]
        patch[r, c] = color


def _paint_checker(patch: np.ndarray, color: int) -> None:
    h, w = patch.shape
    r0 = _randint(1, h - 3)
    c0 = _randint(1, w - 3)
    rh = _randint(2, min(4, h - r0 - 1))
    rw = _randint(2, min(4, w - c0 - 1))
    sub = patch[r0 : r0 + rh, c0 : c0 + rw]
    rr, cc = np.indices(sub.shape)
    sub[(rr + cc) % 2 == 0] = color


def _decorate_master(patch: np.ndarray, base_color: int, deco_colors: list[int]) -> None:
    assert patch.shape[0] >= 6 and patch.shape[1] >= 6
    attempts = 0
    while np.all(patch == base_color):
        attempts += 1
        shape_count = _randint(2, 5)
        for _ in range(shape_count):
            color = deco_colors[_randint(0, len(deco_colors) - 1)]
            shape = _DECORATION_SHAPES[_randint(0, len(_DECORATION_SHAPES) - 1)]
            if shape == "rect":
                _paint_rect(patch, color)
            elif shape == "hline":
                _paint_hline(patch, color)
            elif shape == "vline":
                _paint_vline(patch, color)
            elif shape == "plus":
                _paint_plus(patch, color)
            elif shape == "diag":
                _paint_diag(patch, color)
            elif shape == "dots":
                _paint_dots(patch, color)
            else:
                _paint_checker(patch, color)
        patch[0, :] = base_color
        patch[-1, :] = base_color
        patch[:, 0] = base_color
        patch[:, -1] = base_color
        if attempts > 10:
            patch[patch.shape[0] // 2, patch.shape[1] // 2] = deco_colors[0]


def _box_overlap(a: tuple[int, int, int], b: tuple[int, int, int], pad: int = 0) -> bool:
    ar0, ac0, asz = a
    br0, bc0, bsz = b
    ar1 = ar0 + asz - 1
    ac1 = ac0 + asz - 1
    br1 = br0 + bsz - 1
    bc1 = bc0 + bsz - 1
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _reflect_from_target_axis(target0: int, size: int, master0: int, master_size: int, mode: int) -> int:
    master1_exclusive = master0 + master_size
    if mode < 0:
        return 2 * master0 - target0 - size
    if mode > 0:
        return 2 * master1_exclusive - target0 - size
    return target0


def _target_candidates(
    grid: np.ndarray,
    master_box: tuple[int, int, int],
    target_boxes: list[tuple[int, int, int]],
    mark_boxes: list[tuple[int, int, int]],
    decorated_mask: np.ndarray,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    h, w = grid.shape
    mr0, mc0, master_size = master_box
    out: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    size_max = min(5, master_size - 2)
    for size in np.random.permutation(np.arange(2, size_max + 1)):
        size = int(size)
        row_hi = mr0 + master_size - size - 1
        col_hi = mc0 + master_size - size - 1
        if row_hi < mr0 + 1 or col_hi < mc0 + 1:
            continue
        row_choices = np.random.permutation(np.arange(mr0 + 1, row_hi + 1))
        col_choices = np.random.permutation(np.arange(mc0 + 1, col_hi + 1))
        for tr0 in row_choices:
            for tc0 in col_choices:
                target_box = (int(tr0), int(tc0), size)
                if any(_box_overlap(target_box, other) for other in target_boxes):
                    continue
                for dr, dc in _DIRECTIONS:
                    sr0 = _reflect_from_target_axis(int(tr0), size, mr0, master_size, dr)
                    sc0 = _reflect_from_target_axis(int(tc0), size, mc0, master_size, dc)
                    mark_box = (int(sr0), int(sc0), size)
                    if sr0 < 0 or sc0 < 0 or sr0 + size > h or sc0 + size > w:
                        continue
                    if _box_overlap(mark_box, master_box):
                        continue
                    if any(_box_overlap(mark_box, other, pad=1) for other in mark_boxes):
                        continue
                    rich = int(
                        np.any(
                            decorated_mask[
                                int(tr0) - mr0 : int(tr0) - mr0 + size,
                                int(tc0) - mc0 : int(tc0) - mc0 + size,
                            ]
                        )
                    )
                    out.append((target_box, mark_box, rich))
    np.random.shuffle(out)
    out.sort(key=lambda item: item[2], reverse=True)
    return [(target_box, mark_box) for target_box, mark_box, _rich in out]


def _build_expected(grid: np.ndarray, bg: int, plans: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> np.ndarray:
    out = grid.copy()
    original = grid.copy()
    for target_box, mark_box in plans:
        tr0, tc0, size = target_box
        sr0, sc0, _ = mark_box
        source = original[tr0 : tr0 + size, tc0 : tc0 + size].copy()
        out[sr0 : sr0 + size, sc0 : sc0 + size] = source
    for target_box, _mark_box in plans:
        tr0, tc0, size = target_box
        out[tr0 : tr0 + size, tc0 : tc0 + size] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    bg, base_color, deco_colors = _sample_colors()
    master_size = _randint(_MASTER_MIN, _MASTER_MAX)
    height = _randint(max(_GRID_MIN, master_size + 8), _GRID_MAX)
    width = _randint(max(_GRID_MIN, master_size + 8), _GRID_MAX)
    if height <= master_size + 2 or width <= master_size + 2:
        return None

    mr0 = _randint(2, height - master_size - 3)
    mc0 = _randint(2, width - master_size - 3)
    grid = np.full((height, width), bg, dtype=int)
    grid[mr0 : mr0 + master_size, mc0 : mc0 + master_size] = base_color
    master_patch = grid[mr0 : mr0 + master_size, mc0 : mc0 + master_size]
    _decorate_master(master_patch, base_color, deco_colors)
    decorated_mask = master_patch != base_color

    plan_count = _randint(*_MARK_COUNT_RANGE)
    target_boxes: list[tuple[int, int, int]] = []
    mark_boxes: list[tuple[int, int, int]] = []
    plans: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    master_box = (mr0, mc0, master_size)

    for _ in range(plan_count):
        candidates = _target_candidates(grid, master_box, target_boxes, mark_boxes, decorated_mask)
        if not candidates:
            return None
        target_box, mark_box = candidates[0]
        tr0, tc0, size = target_box
        sr0, sc0, _ = mark_box
        grid[sr0 : sr0 + size, sc0 : sc0 + size] = base_color
        target_boxes.append(target_box)
        mark_boxes.append(mark_box)
        plans.append((target_box, mark_box))

    expected = _build_expected(grid, bg, plans)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
