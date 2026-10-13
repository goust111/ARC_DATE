import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 6]
GEN_ATTEMPTS = 500


@dataclass
class ArcObject:
    color: int
    mask: np.ndarray
    cells: np.ndarray
    r0: int
    c0: int
    sym_h: bool
    sym_v: bool

    @property
    def h(self) -> int:
        return int(self.mask.shape[0])

    @property
    def w(self) -> int:
        return int(self.mask.shape[1])

    @property
    def area(self) -> int:
        return int(self.cells.shape[0])

    @property
    def patch(self) -> np.ndarray:
        patch = np.zeros((self.h, self.w), dtype=int)
        patch[self.mask] = int(self.color)
        return patch

    @property
    def symmetric(self) -> bool:
        return bool(self.sym_h or self.sym_v)


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SYMMETRIC_POOLS = {
    "v_only": [
        _mask(["010", "111"]),
        _mask(["010", "111", "010", "010"]),
        _mask(["010", "111", "101"]),
    ],
    "h_only": [
        _mask(["01", "11", "01"]),
        _mask(["100", "111", "100"]),
        _mask(["011", "110", "011"]),
    ],
    "both": [
        _mask(["11", "11"]),
        _mask(["010", "111", "010"]),
        _mask(["111", "101", "111"]),
    ],
}

ASYMMETRIC_POOL = [
    _mask(["10", "10", "11"]),
    _mask(["110", "011"]),
    _mask(["111", "100"]),
    _mask(["100", "111", "001"]),
    _mask(["110", "100", "111"]),
    _mask(["010", "111", "001"]),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.ravel(grid))).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


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
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _symmetry_axes(mask: np.ndarray) -> tuple[bool, bool]:
    return bool(np.array_equal(mask, mask[::-1, :])), bool(np.array_equal(mask, mask[:, ::-1]))


def _make_object(color: int, mask: np.ndarray, r0: int, c0: int) -> ArcObject:
    sym_h, sym_v = _symmetry_axes(mask)
    return ArcObject(
        color=int(color),
        mask=mask.copy(),
        cells=np.argwhere(mask),
        r0=int(r0),
        c0=int(c0),
        sym_h=sym_h,
        sym_v=sym_v,
    )


def _extract_objects(grid: np.ndarray) -> list[ArcObject]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects: list[ArcObject] = []
    for color in sorted({int(v) for v in np.ravel(g) if int(v) != bg}):
        for coords in _components(g == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            objects.append(_make_object(color, mask, int(r0), int(c0)))
    return sorted(objects, key=lambda o: (o.r0, o.c0, o.color))


def _can_place(occupied: np.ndarray, obj: ArcObject, c0: int) -> bool:
    if c0 < 0 or c0 + obj.w > occupied.shape[1]:
        return False
    rr = obj.cells[:, 0] + obj.r0
    cc = obj.cells[:, 1] + int(c0)
    return not occupied[rr, cc].any()


def _stamp(occupied: np.ndarray, out: np.ndarray, obj: ArcObject, c0: int) -> None:
    rr = obj.cells[:, 0] + obj.r0
    cc = obj.cells[:, 1] + int(c0)
    occupied[rr, cc] = True
    out[rr, cc] = int(obj.color)


def _settle_group(
    objects: list[ArcObject],
    shape: tuple[int, int],
    move_left: bool,
) -> tuple[np.ndarray, dict[int, int], dict[int, bool]]:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=int)
    final_c: dict[int, int] = {}
    blocked: dict[int, bool] = {}
    if move_left:
        ordered = sorted(objects, key=lambda o: (o.c0, o.r0, o.color))
    else:
        ordered = sorted(objects, key=lambda o: (-(o.c0 + o.w - 1), o.r0, o.color))

    for obj in ordered:
        cols = range(0, w - obj.w + 1) if move_left else range(w - obj.w, -1, -1)
        for c0 in cols:
            if _can_place(occupied, obj, c0):
                _stamp(occupied, out, obj, c0)
                final_c[obj.color] = int(c0)
                boundary_c = 0 if move_left else w - obj.w
                blocked[obj.color] = bool(c0 != boundary_c)
                break
        else:
            raise AssertionError("object could not be settled")
    return out, final_c, blocked


def _solve_objects(objects: list[ArcObject], shape: tuple[int, int]) -> tuple[np.ndarray, dict[int, int], dict[int, bool]]:
    left = [obj for obj in objects if obj.symmetric]
    right = [obj for obj in objects if not obj.symmetric]
    assert left and right, "expected both symmetric and asymmetric objects"
    left_out, left_cols, left_blocked = _settle_group(left, shape, move_left=True)
    right_out, right_cols, right_blocked = _settle_group(right, shape, move_left=False)
    overlap = (left_out != BG) & (right_out != BG)
    assert not overlap.any(), "left and right groups must remain separated"
    out = left_out.copy()
    out[right_out != BG] = right_out[right_out != BG]
    final_cols = {**left_cols, **right_cols}
    blocked = {**left_blocked, **right_blocked}
    return out, final_cols, blocked


def _solve_by_predicate(
    objects: list[ArcObject],
    shape: tuple[int, int],
    move_left_pred,
) -> np.ndarray | None:
    left = [obj for obj in objects if move_left_pred(obj)]
    right = [obj for obj in objects if not move_left_pred(obj)]
    if not left or not right:
        return None
    try:
        left_out, _, _ = _settle_group(left, shape, move_left=True)
        right_out, _, _ = _settle_group(right, shape, move_left=False)
    except AssertionError:
        return None
    if ((left_out != BG) & (right_out != BG)).any():
        return None
    out = left_out.copy()
    out[right_out != BG] = right_out[right_out != BG]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide axis-symmetric objects left and fully asymmetric objects right, stacking within each class.

    Input structure:
    - The grid has a black background and several separated monochromatic 4-connected objects.
    - Each object fits in a small bounding box, keeps a single color, and is uniquely identifiable by shape and position.
    - Some objects have a horizontal or vertical mirror axis inside their own bounding box; the rest have neither axis.

    Rule:
    - Detect every foreground object and test its cropped binary shape for horizontal and vertical mirror symmetry.
    - Objects with at least one symmetry axis keep their row coordinates and slide left until the left edge or an earlier left-settled object blocks them.
    - Objects with no symmetry axis keep their row coordinates and slide right until the right edge or an earlier right-settled object blocks them.
    - Left-movers are processed by increasing original left edge, right-movers by decreasing original right edge; shape, color, and orientation never change.

    Key insight:
    - Mirror symmetry is only a routing label: it decides which horizontal packing zone an object belongs to, and the later stacking depends on that classification and on the original within-zone order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _extract_objects(g)
    assert objects, "expected at least one foreground object"
    out, _, _ = _solve_objects(objects, g.shape)
    return out


def _split_nonnegative(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    seq = []
    prev = -1
    end = total + parts - 1
    for cut in cuts + [end]:
        seq.append(cut - prev - 1)
        prev = cut
    return [int(x) for x in seq]


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _sample_overlapping_rows(heights: list[int], h: int) -> list[int] | None:
    rows = [random.randint(0, h - heights[0])]
    candidates = [
        r for r in range(0, h - heights[1] + 1)
        if _intervals_overlap(rows[0], rows[0] + heights[0] - 1, r, r + heights[1] - 1)
    ]
    if not candidates:
        return None
    rows.append(random.choice(candidates))
    for idx in range(2, len(heights)):
        overlap_rows = [
            r for r in range(0, h - heights[idx] + 1)
            if any(
                _intervals_overlap(rows[j], rows[j] + heights[j] - 1, r, r + heights[idx] - 1)
                for j in range(idx)
            )
        ]
        if not overlap_rows:
            return None
        rows.append(random.choice(overlap_rows if random.random() < 0.8 else list(range(0, h - heights[idx] + 1))))
    return rows


def _compose_columns(widths: list[int], w: int) -> list[int]:
    base = sum(widths) + 2 * (len(widths) - 1) + 2
    spare = w - base
    extras = _split_nonnegative(spare, len(widths) + 1)
    col = 1 + extras[0]
    cols = [col]
    for idx in range(1, len(widths)):
        col += widths[idx - 1] + 2 + extras[idx]
        cols.append(col)
    return cols


def _left_right_gap(objects: list[ArcObject], final_cols: dict[int, int]) -> int:
    left_edges = [final_cols[obj.color] + obj.w - 1 for obj in objects if obj.symmetric]
    right_edges = [final_cols[obj.color] for obj in objects if not obj.symmetric]
    return int(min(right_edges) - max(left_edges) - 1)


def _areas_overlap(left_masks: list[np.ndarray], right_masks: list[np.ndarray]) -> bool:
    left_areas = {int(mask.sum()) for mask in left_masks}
    right_areas = {int(mask.sum()) for mask in right_masks}
    return bool(left_areas & right_areas)


def _pick_symmetric_masks(left_count: int) -> list[np.ndarray]:
    masks = [
        random.choice(SYMMETRIC_POOLS["v_only"]).copy(),
        random.choice(SYMMETRIC_POOLS["h_only"]).copy(),
    ]
    if left_count == 3:
        extra_pool = (
            SYMMETRIC_POOLS["both"] if random.random() < 0.65
            else SYMMETRIC_POOLS[random.choice(["v_only", "h_only"])]
        )
        masks.append(random.choice(extra_pool).copy())
    random.shuffle(masks)
    return masks


def _pick_asymmetric_masks(match_area: int) -> list[np.ndarray] | None:
    by_area: dict[int, list[np.ndarray]] = {}
    for mask in ASYMMETRIC_POOL:
        by_area.setdefault(int(mask.sum()), []).append(mask)
    if match_area not in by_area:
        return None
    first = random.choice(by_area[match_area]).copy()
    second = random.choice(ASYMMETRIC_POOL).copy()
    return [first, second]


def _validate_shape_pools() -> None:
    for name, masks in SYMMETRIC_POOLS.items():
        for mask in masks:
            assert 4 <= int(mask.sum()) <= 8
            assert max(mask.shape) <= 4
            assert len(_components(mask)) == 1
            sym_h, sym_v = _symmetry_axes(mask)
            if name == "v_only":
                assert sym_v and not sym_h
            elif name == "h_only":
                assert sym_h and not sym_v
            else:
                assert sym_h and sym_v
    for mask in ASYMMETRIC_POOL:
        assert 4 <= int(mask.sum()) <= 8
        assert max(mask.shape) <= 4
        assert len(_components(mask)) == 1
        sym_h, sym_v = _symmetry_axes(mask)
        assert not sym_h and not sym_v


_validate_shape_pools()


def _attempt() -> np.ndarray | None:
    left_count = random.choice([2, 3])
    sym_masks = _pick_symmetric_masks(left_count)
    matchable = [int(mask.sum()) for mask in sym_masks if int(mask.sum()) in {int(m.sum()) for m in ASYMMETRIC_POOL}]
    if not matchable:
        return None
    asym_masks = _pick_asymmetric_masks(random.choice(matchable))
    if asym_masks is None:
        return None

    masks = sym_masks + asym_masks
    n = len(masks)
    heights = [int(mask.shape[0]) for mask in masks]
    widths = [int(mask.shape[1]) for mask in masks]
    h_lo = max(14, max(heights) + 5)
    h_hi = 18
    if h_lo > h_hi:
        return None
    h = random.randint(h_lo, h_hi)

    left_rows = _sample_overlapping_rows(heights[:left_count], h)
    right_rows = _sample_overlapping_rows(heights[left_count:], h)
    if left_rows is None or right_rows is None:
        return None
    left_bottom = max(r + hh - 1 for r, hh in zip(left_rows, heights[:left_count]))
    right_bottom = max(r + hh - 1 for r, hh in zip(right_rows, heights[left_count:]))
    if left_bottom < min(right_rows) or right_bottom < min(left_rows):
        return None

    order = list(range(n))
    random.shuffle(order)
    ordered_widths = [widths[i] for i in order]
    w_lo = max(20, sum(ordered_widths) + 2 * (n - 1) + 2)
    w_hi = 30
    if w_lo > w_hi:
        return None
    w = random.randint(w_lo, w_hi)
    cols_by_index = {}
    for idx, c0 in zip(order, _compose_columns(ordered_widths, w)):
        cols_by_index[idx] = int(c0)

    colors = random.sample(PALETTE, n)
    objects: list[ArcObject] = []
    for idx in range(left_count):
        objects.append(_make_object(colors[idx], sym_masks[idx], left_rows[idx], cols_by_index[idx]))
    for jdx in range(2):
        idx = left_count + jdx
        objects.append(_make_object(colors[idx], asym_masks[jdx], right_rows[jdx], cols_by_index[idx]))

    if not _areas_overlap(sym_masks, asym_masks):
        return None

    out, final_cols, blocked = _solve_objects(objects, (h, w))
    if _left_right_gap(objects, final_cols) < 2:
        return None
    if sum(blocked[obj.color] for obj in objects if obj.symmetric) < 1:
        return None
    if sum(blocked[obj.color] for obj in objects if not obj.symmetric) < 1:
        return None

    moved_left = any(final_cols[obj.color] < obj.c0 for obj in objects if obj.symmetric)
    moved_right = any(final_cols[obj.color] > obj.c0 for obj in objects if not obj.symmetric)
    if not (moved_left and moved_right):
        return None

    inp = np.zeros((h, w), dtype=int)
    for obj in objects:
        rr = obj.cells[:, 0] + obj.r0
        cc = obj.cells[:, 1] + obj.c0
        inp[rr, cc] = int(obj.color)

    extracted = _extract_objects(inp)
    if len(extracted) != n:
        return None
    if {obj.color for obj in extracted} != {obj.color for obj in objects}:
        return None
    if any(obj.c0 == 0 or obj.c0 + obj.w == w for obj in objects):
        return None

    true_out = main(inp)
    if not np.array_equal(true_out, out):
        return None
    if np.array_equal(inp, out):
        return None

    alt_preds = [
        lambda obj: True,
        lambda obj: False,
        lambda obj: obj.sym_h,
        lambda obj: obj.sym_v,
        lambda obj: obj.sym_h and obj.sym_v,
    ]
    for pred in alt_preds:
        alt = _solve_by_predicate(objects, (h, w), pred)
        if alt is not None and np.array_equal(alt, out):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-split stacking input")
