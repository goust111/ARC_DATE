from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, find_components  # noqa: E402


BG = Color.BLACK
SRC = Color.GREY
ACTIVE = Color.TEAL
RAY = Color.YELLOW
OVERLAP = Color.PINK
MIN_GAP = 3
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
GEN_RETRIES = 400


def _centered_span(lo: int, hi: int, length: int) -> tuple[int, int]:
    center2 = int(lo) + int(hi)
    start = (center2 - (int(length) - 1)) // 2
    end = start + int(length) - 1
    return int(start), int(end)


def _normalize_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    pts[:, 0] -= int(pts[:, 0].min())
    pts[:, 1] -= int(pts[:, 1].min())
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _mirror_lr(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(mask).copy()


def _active_pixel_chain(length: int) -> np.ndarray:
    return _normalize_mask([(i, i) for i in range(int(length))])


def _active_hdomino_chain(length: int) -> np.ndarray:
    coords = []
    for i in range(int(length)):
        coords.append((i, 2 * i))
        coords.append((i, 2 * i + 1))
    return _normalize_mask(coords)


def _active_cycle4() -> np.ndarray:
    coords = [
        (0, 1), (0, 2),
        (1, 0), (2, 0),
        (1, 3), (2, 3),
        (3, 1), (3, 2),
    ]
    return _normalize_mask(coords)


def _inert_rect(height: int, width: int) -> np.ndarray:
    return np.ones((int(height), int(width)), dtype=bool)


def _inert_branch() -> np.ndarray:
    corners = random.sample([(0, 0), (0, 2), (2, 0), (2, 2)], k=random.choice((3, 4)))
    return _normalize_mask([(1, 1), *corners])


def _inert_stair() -> np.ndarray:
    coords = [(0, 0), (0, 1), (1, 1), (1, 2)]
    if random.random() < 0.5:
        coords.append((2, 2))
    return _normalize_mask(coords)


def _shape_box(top: int, left: int, mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _boxes_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_GAP) -> bool:
    row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    if row_gap == 0 and col_gap == 0:
        return False
    return max(row_gap, col_gap) >= int(gap)


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int = SRC) -> None:
    rows, cols = np.nonzero(mask)
    grid[rows + int(top), cols + int(left)] = int(color)


def _crop_region_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_diagonal_chain(region_mask: np.ndarray) -> bool:
    lobes = components_mask(region_mask, connectivity=4)
    if len(lobes) < 2:
        return False

    label_map = -np.ones(region_mask.shape, dtype=int)
    for idx, coords in enumerate(lobes):
        label_map[coords[:, 0], coords[:, 1]] = idx

    neighbors = [set() for _ in lobes]
    h, w = region_mask.shape
    for idx, coords in enumerate(lobes):
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in DIAGONALS:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(label_map[nr, nc])
                    if other >= 0 and other != idx:
                        neighbors[idx].add(other)

    if any(len(adj) > 2 for adj in neighbors):
        return False

    seen = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(neighbors[node] - seen)
    if len(seen) != len(lobes):
        return False

    edge_count = sum(len(adj) for adj in neighbors) // 2
    return edge_count in {len(lobes) - 1, len(lobes)}


def _analyze_regions(grid: np.ndarray) -> list[dict]:
    regions = find_components(
        grid,
        bg=BG,
        colors=[SRC],
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    out = []
    for region in regions:
        bbox = region["bbox"]
        local_mask = _crop_region_mask(region["coords"], bbox)
        out.append(
            {
                "bbox": bbox,
                "coords": region["coords"],
                "active": _is_diagonal_chain(local_mask),
            }
        )
    return out


def _collect_direction_cells(
    bbox: tuple[int, int, int, int],
    inert_mask: np.ndarray,
    direction: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []

    if direction in {"top", "bottom"}:
        base = c1 - c0 + 1
        step = 1
        while True:
            length = base - 2 * (step - 1)
            if length < 1:
                break
            row = r0 - step if direction == "top" else r1 + step
            lo, hi = _centered_span(c0, c1, length)
            span = [(row, col) for col in range(lo, hi + 1)]
            if any(
                not (0 <= rr < inert_mask.shape[0] and 0 <= cc < inert_mask.shape[1]) or inert_mask[rr, cc]
                for rr, cc in span
            ):
                break
            cells.extend(span)
            step += 1
    else:
        base = r1 - r0 + 1
        step = 1
        while True:
            length = base - 2 * (step - 1)
            if length < 1:
                break
            col = c0 - step if direction == "left" else c1 + step
            lo, hi = _centered_span(r0, r1, length)
            span = [(row, col) for row in range(lo, hi + 1)]
            if any(
                not (0 <= rr < inert_mask.shape[0] and 0 <= cc < inert_mask.shape[1]) or inert_mask[rr, cc]
                for rr, cc in span
            ):
                break
            cells.extend(span)
            step += 1

    return cells


def _fan_map(active_boxes: list[tuple[int, int, int, int]], inert_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    coverage = np.zeros(inert_mask.shape, dtype=int)
    by_box: dict[tuple[int, int, int, int], dict[str, list[tuple[int, int]]]] = {}
    for bbox in active_boxes:
        by_dir = {}
        for direction in ("top", "bottom", "left", "right"):
            coords = _collect_direction_cells(bbox, inert_mask, direction)
            by_dir[direction] = coords
            for r, c in coords:
                coverage[r, c] += 1
        by_box[bbox] = by_dir
    return coverage, by_box


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BG, SRC}

    analysis = _analyze_regions(g)
    inert_mask = np.zeros(g.shape, dtype=bool)
    active_boxes = []
    out = g.copy()
    for region in analysis:
        coords = region["coords"]
        rr = coords[:, 0]
        cc = coords[:, 1]
        if region["active"]:
            out[rr, cc] = ACTIVE
            active_boxes.append(region["bbox"])
        else:
            inert_mask[rr, cc] = True

    coverage, _ = _fan_map(active_boxes, inert_mask)
    out[coverage == 1] = RAY
    out[coverage >= 2] = OVERLAP
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor every diagonal-chain grey region cyan and let only those active regions emit shrinking yellow rays.

    Input structure:
    - The grid is a black background containing several grey 8-connected regions.
    - Each grey region splits into one or more 4-connected lobes inside its own bounding box.
    - Some regions form diagonal-only lobe chains or rings, while others contain a branch or a single bulky 4-connected block.

    Rule:
    - Classify each grey 8-connected region by its 4-connected lobe graph: active regions are exactly the connected path-or-cycle cases where every lobe touches at most two other lobes diagonally.
    - Recolor active regions to cyan and keep inactive regions grey.
    - From every active region's bounding box, emit four outward shrinking fans whose first run matches the side length, then shrinks by two per step; a fan direction stops before painting a run if that run would hit an inactive grey cell or leave the grid.
    - Paint singly covered fan pixels yellow and pixels covered by multiple active fans magenta.

    Key insight:
    - The same diagonal-chain test controls three visible consequences at once: which regions turn cyan, which regions are allowed to emit rays, and which grey regions remain behind as ray blockers.
    """
    return _solve(input_grid)


def _fan_step_count(length: int) -> int:
    return (int(length) + 1) // 2


def _choose_active_b(shared_h: int) -> np.ndarray:
    options = [_active_pixel_chain(shared_h), _active_hdomino_chain(shared_h)]
    if shared_h == 4:
        options.append(_active_cycle4())
    mask = options[random.randrange(len(options))]
    return _mirror_lr(mask) if random.random() < 0.5 else mask


def _choose_extra_shape() -> tuple[np.ndarray, bool]:
    if random.random() < 0.5:
        family = random.choice(
            [
                _active_pixel_chain(random.choice((4, 5, 6))),
                _active_hdomino_chain(random.choice((3, 4))),
                _active_cycle4(),
            ]
        )
        return (_mirror_lr(family) if random.random() < 0.5 else family), True

    inert = random.choice(
        [
            _inert_branch(),
            _inert_stair() if random.random() < 0.5 else _mirror_lr(_inert_stair()),
            _inert_rect(*random.choice(((2, 2), (2, 3), (3, 2)))),
        ]
    )
    return inert, False


def _find_spot(
    canvas: tuple[int, int],
    mask: np.ndarray,
    placed: list[tuple[int, int, int, int]],
    row_pref: list[int] | None = None,
    col_pref: list[int] | None = None,
) -> tuple[int, int] | None:
    h, w = mask.shape
    H, W = canvas
    rows = list(range(0, H - h + 1)) if row_pref is None else [r for r in row_pref if 0 <= r <= H - h]
    cols = list(range(0, W - w + 1)) if col_pref is None else [c for c in col_pref if 0 <= c <= W - w]
    if row_pref is None:
        random.shuffle(rows)
    if col_pref is None:
        random.shuffle(cols)

    for r in rows:
        trial_cols = cols.copy()
        if col_pref is None:
            random.shuffle(trial_cols)
        for c in trial_cols:
            box = _shape_box(r, c, mask)
            if all(_boxes_far(box, other) for other in placed):
                return int(r), int(c)
    return None


def _semantic_ok(grid: np.ndarray, expected_flags: dict[tuple[int, int, int, int], bool], blocker_box: tuple[int, int, int, int], overlap_expected: bool) -> bool:
    analysis = _analyze_regions(grid)
    if not (4 <= len(analysis) <= 7):
        return False

    actual_flags = {tuple(region["bbox"]): bool(region["active"]) for region in analysis}
    if actual_flags != expected_flags:
        return False

    active_boxes = [bbox for bbox, active in actual_flags.items() if active]
    inert_boxes = [bbox for bbox, active in actual_flags.items() if not active]
    if len(active_boxes) < 2 or len(inert_boxes) < 2:
        return False

    for i, box_a in enumerate(actual_flags):
        for box_b in list(actual_flags)[i + 1 :]:
            if not _boxes_far(box_a, box_b):
                return False

    inert_mask = np.zeros(grid.shape, dtype=bool)
    for region in analysis:
        if not region["active"]:
            coords = region["coords"]
            inert_mask[coords[:, 0], coords[:, 1]] = True

    coverage, by_box = _fan_map(active_boxes, inert_mask)
    if overlap_expected and int((coverage >= 2).sum()) == 0:
        return False
    if int((coverage == 1).sum()) == 0:
        return False

    blocker_hit = False
    for bbox, dirs in by_box.items():
        if bbox[1] <= blocker_box[3] and blocker_box[1] <= bbox[3]:
            theoretical = _fan_step_count(bbox[3] - bbox[1] + 1)
            if len(dirs["top"]) < sum(max(0, (bbox[3] - bbox[1] + 1) - 2 * i) for i in range(theoretical)):
                blocker_hit = True
                break
    if not blocker_hit:
        return False

    expected = _solve(grid)
    if not np.array_equal(main(grid), expected):
        return False

    colors = set(map(int, np.unique(expected)))
    if not {ACTIVE, SRC, RAY, OVERLAP}.issubset(colors):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(20, 26)
    W = random.randint(22, 28)
    shared_h = random.choice((4, 5))

    mask_a = _active_hdomino_chain(shared_h)
    if random.random() < 0.5:
        mask_a = _mirror_lr(mask_a)
    mask_b = _choose_active_b(shared_h)

    mask_c = _inert_rect(*random.choice(((2, 2), (2, 3), (3, 2))))
    mask_d = random.choice([_inert_branch(), _inert_stair(), _inert_rect(2, 2)])
    if random.random() < 0.5:
        mask_d = _mirror_lr(mask_d)

    h_a, w_a = mask_a.shape
    h_b, w_b = mask_b.shape
    h_c, _w_c = mask_c.shape
    top_steps_a = _fan_step_count(w_a)
    side_steps = _fan_step_count(shared_h)
    if top_steps_a < 4:
        return None

    max_gap = min(5, 2 * side_steps - 1)
    if max_gap < MIN_GAP:
        return None
    gap_ab = random.randint(MIN_GAP, max_gap)
    gap_up = random.randint(MIN_GAP, min(4, top_steps_a - 1))

    total_width = w_a + gap_ab + w_b
    if total_width + 2 > W:
        return None

    min_row = h_c + gap_up
    max_row = H - max(h_a, h_b) - 4
    if min_row > max_row:
        return None
    row_ab = random.randint(min_row, max_row)
    col_a = random.randint(1, W - total_width - 1)
    col_b = col_a + w_a + gap_ab

    box_a = _shape_box(row_ab, col_a, mask_a)
    box_b = _shape_box(row_ab, col_b, mask_b)

    blocked_len = w_a - 2 * gap_up
    if blocked_len < 1:
        return None
    blocked_lo, blocked_hi = _centered_span(col_a, col_a + w_a - 1, blocked_len)
    row_c = row_ab - h_c - gap_up
    span_center = (blocked_lo + blocked_hi) // 2
    col_c = max(0, min(W - mask_c.shape[1], span_center - mask_c.shape[1] // 2))
    box_c = _shape_box(row_c, col_c, mask_c)

    if not all(_boxes_far(box_c, other) for other in (box_a, box_b)):
        return None

    placed_boxes = [box_a, box_b, box_c]
    pos_d = _find_spot((H, W), mask_d, placed_boxes, row_pref=list(range(H - mask_d.shape[0], -1, -1)))
    if pos_d is None:
        return None
    row_d, col_d = pos_d
    box_d = _shape_box(row_d, col_d, mask_d)
    placed_boxes.append(box_d)

    shapes: list[tuple[np.ndarray, tuple[int, int, int, int], bool]] = [
        (mask_a, box_a, True),
        (mask_b, box_b, True),
        (mask_c, box_c, False),
        (mask_d, box_d, False),
    ]

    target_total = random.randint(4, 7)
    while len(shapes) < target_total:
        extra_mask, extra_active = _choose_extra_shape()
        pos = _find_spot((H, W), extra_mask, placed_boxes)
        if pos is None:
            break
        row_e, col_e = pos
        box_e = _shape_box(row_e, col_e, extra_mask)
        placed_boxes.append(box_e)
        shapes.append((extra_mask, box_e, extra_active))

    grid = np.full((H, W), BG, dtype=int)
    expected_flags: dict[tuple[int, int, int, int], bool] = {}
    for mask, box, active in shapes:
        _place_mask(grid, mask, box[0], box[1], SRC)
        expected_flags[box] = active

    if not _semantic_ok(grid, expected_flags, box_c, overlap_expected=True):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-chain ray puzzle input")
