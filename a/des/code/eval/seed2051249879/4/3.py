from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_contiguous_run(mask_1d: np.ndarray) -> bool:
    idx = np.flatnonzero(mask_1d)
    return bool(idx.size) and int(idx[-1] - idx[0] + 1) == int(idx.size)


def _frame_info_from_component(coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(local & ~border):
        return None
    if not (local[0, 0] and local[0, -1] and local[-1, 0] and local[-1, -1]):
        return None

    missing_top = ~local[0, 1:-1]
    missing_bottom = ~local[-1, 1:-1]
    missing_left = ~local[1:-1, 0]
    missing_right = ~local[1:-1, -1]

    sides = {
        "top": missing_top,
        "bottom": missing_bottom,
        "left": missing_left,
        "right": missing_right,
    }
    gap_sides = [name for name, arr in sides.items() if np.any(arr)]
    if len(gap_sides) != 1:
        return None

    gap_side = gap_sides[0]
    gap_mask = sides[gap_side]
    if not _is_contiguous_run(gap_mask):
        return None

    for name, arr in sides.items():
        if name != gap_side and np.any(arr):
            return None

    gap_idx = np.flatnonzero(gap_mask)
    gap_start = int(gap_idx[0] + 1)
    gap_len = int(gap_idx.size)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_side": gap_side,
        "gap_start": gap_start,
        "gap_len": gap_len,
        "coords": coords,
        "area_key": (h - 2) * (w - 2),
    }


def _find_frame_for_color(grid: np.ndarray, color: int) -> dict | None:
    candidates = []
    for comp in _components(grid == color):
        info = _frame_info_from_component(comp)
        if info is not None:
            candidates.append(info)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["area_key"], len(item["coords"])), reverse=True)
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather same-color fragments back into the matching gapped rectangular frame.

    The final state is always the same once the frame is identified:
    erase same-color fragments outside the frame and fill the entire frame interior.
    The missing border segment remains background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        frame = _find_frame_for_color(g, color)
        if frame is None:
            continue

        frame_mask = np.zeros_like(g, dtype=bool)
        for r, c in frame["coords"]:
            frame_mask[r, c] = True

        out[(g == color) & (~frame_mask)] = bg
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = color

    return out


def _draw_gapped_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    gap_side: str,
    gap_start: int,
    gap_len: int,
) -> list[tuple[int, int]]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    frame_cells: list[tuple[int, int]] = []

    def add(rr: int, cc: int) -> None:
        grid[rr, cc] = color
        frame_cells.append((rr, cc))

    gap_set: set[tuple[int, int]] = set()
    if gap_side == "top":
        for offset in range(gap_len):
            gap_set.add((r0, c0 + gap_start + offset))
    elif gap_side == "bottom":
        for offset in range(gap_len):
            gap_set.add((r1, c0 + gap_start + offset))
    elif gap_side == "left":
        for offset in range(gap_len):
            gap_set.add((r0 + gap_start + offset, c0))
    else:
        for offset in range(gap_len):
            gap_set.add((r0 + gap_start + offset, c1))

    for rr in range(r0, r1 + 1):
        for cc in (c0, c1):
            if (rr, cc) not in gap_set:
                add(rr, cc)
    for cc in range(c0, c1 + 1):
        for rr in (r0, r1):
            if (rr, cc) not in gap_set and (rr, cc) not in frame_cells:
                add(rr, cc)

    return frame_cells


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _rectangles_intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _partition_rectangle(h: int, w: int, piece_count: int) -> list[np.ndarray] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(80):
        seeds = random.sample(cells, piece_count)
        labels = -np.ones((h, w), dtype=int)
        sizes = [1] * piece_count
        for idx, (r, c) in enumerate(seeds):
            labels[r, c] = idx

        while np.any(labels == -1):
            frontier: dict[int, set[tuple[int, int]]] = {idx: set() for idx in range(piece_count)}
            for r in range(h):
                for c in range(w):
                    label = int(labels[r, c])
                    if label < 0:
                        continue
                    for dr, dc in DIRS4:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                            frontier[label].add((nr, nc))

            growable = [idx for idx, opts in frontier.items() if opts]
            if not growable:
                break

            min_size = min(sizes[idx] for idx in growable)
            preferred = [idx for idx in growable if sizes[idx] <= min_size + 1]
            label = random.choice(preferred)
            nr, nc = random.choice(tuple(frontier[label]))
            labels[nr, nc] = label
            sizes[label] += 1

        if np.any(labels == -1):
            continue
        if min(sizes) < 2:
            continue

        pieces: list[np.ndarray] = []
        for idx in range(piece_count):
            pts = np.argwhere(labels == idx)
            pr0 = int(np.min(pts[:, 0]))
            pc0 = int(np.min(pts[:, 1]))
            pr1 = int(np.max(pts[:, 0]))
            pc1 = int(np.max(pts[:, 1]))
            patch = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
            for r, c in pts:
                patch[int(r) - pr0, int(c) - pc0] = True
            pieces.append(patch)

        if any(not np.any(piece) for piece in pieces):
            continue
        return pieces
    return None


def _mark_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _patch_cells(patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    coords = np.argwhere(patch)
    return [(int(r0 + r), int(c0 + c)) for r, c in coords]


def _candidate_positions(
    blocked: np.ndarray,
    patch: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    gap_side: str,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = patch.shape
    candidates: list[tuple[int, int]] = []
    preferred: list[tuple[int, int]] = []
    fr0, fc0, fr1, fc1 = frame_bbox

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            cells = _patch_cells(patch, r0, c0)
            if any(blocked[r, c] for r, c in cells):
                continue
            candidates.append((r0, c0))

            if gap_side == "top" and r0 + ph - 1 < fr0:
                preferred.append((r0, c0))
            elif gap_side == "bottom" and r0 > fr1:
                preferred.append((r0, c0))
            elif gap_side == "left" and c0 + pw - 1 < fc0:
                preferred.append((r0, c0))
            elif gap_side == "right" and c0 > fc1:
                preferred.append((r0, c0))

    pool = preferred if preferred else candidates
    if not pool:
        return []

    def distance(pos: tuple[int, int]) -> int:
        r0, c0 = pos
        if gap_side == "top":
            return fr0 - (r0 + ph - 1)
        if gap_side == "bottom":
            return r0 - fr1
        if gap_side == "left":
            return fc0 - (c0 + pw - 1)
        return c0 - fc1

    best = min(distance(pos) for pos in pool)
    close = [pos for pos in pool if distance(pos) <= best + 2]
    random.shuffle(close)
    return close


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    color: int,
    frame_bbox: tuple[int, int, int, int],
    gap_side: str,
) -> bool:
    positions = _candidate_positions(blocked, patch, frame_bbox, gap_side)
    if not positions:
        return False
    r0, c0 = random.choice(positions)
    cells = _patch_cells(patch, r0, c0)
    for r, c in cells:
        grid[r, c] = color
    _mark_cells(blocked, cells)
    return True


def _sample_frame_size(object_count: int) -> tuple[int, int]:
    max_side = 9 if object_count <= 2 else 8
    return random.randint(5, max_side), random.randint(5, max_side)


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    bg = random.randint(0, 9)
    object_count = random.randint(1, 3)
    colors = random.sample([c for c in range(10) if c != bg], object_count)

    specs = []
    for color in colors:
        outer_h, outer_w = _sample_frame_size(object_count)
        gap_side = random.choice(["top", "bottom", "left", "right"])
        side_inner = outer_w - 2 if gap_side in {"top", "bottom"} else outer_h - 2
        gap_len = random.randint(1, side_inner)
        gap_start = random.randint(1, side_inner - gap_len + 1)
        specs.append(
            {
                "color": color,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "gap_side": gap_side,
                "gap_start": gap_start,
                "gap_len": gap_len,
            }
        )

    specs.sort(key=lambda item: item["outer_h"] * item["outer_w"], reverse=True)

    frame_bboxes: list[tuple[int, int, int, int]] = []
    for spec in specs:
        placed = False
        for _ in range(100):
            r0 = random.randint(1, h - spec["outer_h"] - 1)
            c0 = random.randint(1, w - spec["outer_w"] - 1)
            bbox = (r0, c0, r0 + spec["outer_h"] - 1, c0 + spec["outer_w"] - 1)
            if any(_rectangles_intersect(bbox, other, pad=2) for other in frame_bboxes):
                continue
            spec["bbox"] = bbox
            frame_bboxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)

    fragment_blocked = np.zeros((h, w), dtype=bool)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        color = spec["color"]
        _draw_gapped_frame(
            inp,
            r0,
            c0,
            spec["outer_h"],
            spec["outer_w"],
            color,
            spec["gap_side"],
            spec["gap_start"],
            spec["gap_len"],
        )
        _draw_gapped_frame(
            expected,
            r0,
            c0,
            spec["outer_h"],
            spec["outer_w"],
            color,
            spec["gap_side"],
            spec["gap_start"],
            spec["gap_len"],
        )
        expected[r0 + 1 : r1, c0 + 1 : c1] = color
        _mark_rect(fragment_blocked, r0, c0, r1, c1, pad=1)

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        area = inner_h * inner_w
        max_pieces = min(5, max(2, area // 3))
        piece_count = random.randint(2, max_pieces)
        pieces = _partition_rectangle(inner_h, inner_w, piece_count)
        if pieces is None:
            return None

        pieces.sort(key=lambda patch: int(np.count_nonzero(patch)), reverse=True)
        for patch in pieces:
            ok = _place_patch(
                inp,
                fragment_blocked,
                patch,
                spec["color"],
                spec["bbox"],
                spec["gap_side"],
            )
            if not ok:
                return None

    if _get_bg(inp) != bg or _get_bg(expected) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _make_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 3")
