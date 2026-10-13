from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("up", "down", "left", "right")
COLOR_CHOICES = tuple(c for c in range(1, 10) if c != GREY)

# The missing corner of the 2x2 marker rotates with the encoded direction.
ARROW_PATTERNS: dict[str, set[tuple[int, int]]] = {
    "up": {(0, 0), (0, 1), (1, 0)},
    "right": {(0, 0), (0, 1), (1, 1)},
    "down": {(0, 1), (1, 0), (1, 1)},
    "left": {(0, 0), (1, 0), (1, 1)},
}

GEN_RETRIES = 500
PLACEMENT_RETRIES = 200
GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
RECT_H_RANGE = (6, 10)
RECT_W_RANGE = (6, 10)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(np.asarray(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    return int(sum(comp.shape[0] for comp in _components(mask))) == int(mask.sum()) and len(_components(mask)) == 1


def _has_full_2x2(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if bool(mask[r : r + 2, c : c + 2].all()):
                return True
    return False


def _frame_bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None
    mask = _mask_from_coords(coords)
    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(mask, expected):
        return None
    return (r0, c0, r1, c1)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _detect_arrow_direction(piece_patch: np.ndarray, piece_mask: np.ndarray) -> str:
    grey_coords = np.argwhere((piece_patch == GREY) & piece_mask)
    if grey_coords.shape[0] != 3:
        raise AssertionError("each piece must contain exactly three grey marker cells")
    r0, c0 = map(int, grey_coords.min(axis=0))
    r1, c1 = map(int, grey_coords.max(axis=0))
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        raise AssertionError("marker must occupy a 2x2 box")
    rel = {(int(r - r0), int(c - c0)) for r, c in grey_coords}
    for side, pattern in ARROW_PATTERNS.items():
        if rel == pattern:
            return side
    raise AssertionError("unknown arrow orientation")


def _extract_scene(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)

    frame_bbox = None
    frame_coords = None
    pieces: list[dict[str, object]] = []

    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        if colors == {GREY}:
            candidate = _frame_bbox_from_coords(coords)
            if candidate is not None:
                assert frame_bbox is None, "expected a unique target frame"
                frame_bbox = candidate
                frame_coords = coords
                continue

        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([r0, c0], dtype=int)
        mask[rel[:, 0], rel[:, 1]] = True
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()

        piece_colors = {int(v) for v in np.unique(patch[mask]) if int(v) != GREY}
        assert len(piece_colors) == 1, "each fragment must have exactly one non-grey color"
        side = _detect_arrow_direction(patch, mask)
        color = next(iter(piece_colors))
        pieces.append(
            {
                "mask": mask,
                "coords": np.argwhere(mask),
                "side": side,
                "color": color,
                "area": int(mask.sum()),
            }
        )

    assert frame_bbox is not None and frame_coords is not None, "expected one grey target frame"
    assert pieces, "expected at least one fragment"
    return frame_bbox, pieces


def _piece_placements(piece: dict[str, object], target_h: int, target_w: int) -> list[dict[str, object]]:
    mask = np.asarray(piece["mask"], dtype=bool)
    side = str(piece["side"])
    ph, pw = mask.shape
    coords = np.asarray(piece["coords"], dtype=int)
    if ph > target_h or pw > target_w:
        return []

    rows: list[int]
    cols: list[int]
    if side == "up":
        if not bool(mask[0, :].any()):
            return []
        rows = [0]
        cols = list(range(target_w - pw + 1))
    elif side == "down":
        if not bool(mask[-1, :].any()):
            return []
        rows = [target_h - ph]
        cols = list(range(target_w - pw + 1))
    elif side == "left":
        if not bool(mask[:, 0].any()):
            return []
        rows = list(range(target_h - ph + 1))
        cols = [0]
    elif side == "right":
        if not bool(mask[:, -1].any()):
            return []
        rows = list(range(target_h - ph + 1))
        cols = [target_w - pw]
    else:
        raise AssertionError(side)

    out: list[dict[str, object]] = []
    for r0 in rows:
        for c0 in cols:
            linear = np.asarray([(r0 + int(r)) * target_w + c0 + int(c) for r, c in coords], dtype=int)
            out.append({"row": int(r0), "col": int(c0), "linear": linear})
    return out


def _solve_cover(
    pieces: list[dict[str, object]],
    target_h: int,
    target_w: int,
    max_solutions: int = 1,
) -> list[list[dict[str, object]]]:
    total_area = sum(int(piece["area"]) for piece in pieces)
    if total_area != target_h * target_w:
        return []

    placements = [_piece_placements(piece, target_h, target_w) for piece in pieces]
    if any(len(opts) == 0 for opts in placements):
        return []

    order = sorted(range(len(pieces)), key=lambda i: (len(placements[i]), -int(pieces[i]["area"])))
    occupied = np.zeros(target_h * target_w, dtype=bool)
    current: list[dict[str, object] | None] = [None] * len(pieces)
    solutions: list[list[dict[str, object]]] = []

    def dfs(depth: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if depth == len(order):
            solutions.append([dict(v) for v in current if v is not None])
            return

        idx = order[depth]
        for placement in placements[idx]:
            linear = np.asarray(placement["linear"], dtype=int)
            if bool(occupied[linear].any()):
                continue
            occupied[linear] = True
            current[idx] = placement
            dfs(depth + 1)
            current[idx] = None
            occupied[linear] = False

    dfs(0)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into the grey target frame.

    Each fragment is a non-black connected component with one 3-cell grey L marker.
    The marker orientation says which side of the target rectangle that fragment must
    touch. Fragments are translated without rotation or reflection, and the grey
    marker cells are restored to the fragment color after placement.
    """
    g = np.asarray(grid, dtype=int)
    frame_bbox, pieces = _extract_scene(g)
    r0, c0, r1, c1 = frame_bbox
    target_h = int(r1 - r0 - 1)
    target_w = int(c1 - c0 - 1)
    assert target_h > 0 and target_w > 0, "target frame must have a non-empty interior"

    solutions = _solve_cover(pieces, target_h, target_w, max_solutions=1)
    assert solutions, "fragments must tile the target rectangle"
    placement_by_piece = solutions[0]

    out = np.full_like(g, BG)
    _draw_frame(out, frame_bbox, GREY)
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    for piece, placement in zip(pieces, placement_by_piece):
        color = int(piece["color"])
        for pr, pc in np.asarray(piece["coords"], dtype=int):
            interior[int(placement["row"]) + int(pr), int(placement["col"]) + int(pc)] = color
    return out


def _profile(rng: random.Random, length: int, low: int, high: int) -> list[int]:
    vals = [rng.randint(low, high) for _ in range(length)]
    if low < high and len(set(vals)) == 1:
        idx = rng.randrange(length)
        vals[idx] = low if vals[idx] > low else high
    return vals


def _partition_two_horizontal(rng: random.Random, h: int, w: int) -> list[dict[str, object]]:
    cut = _profile(rng, w, 2, h - 2)
    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    for c, split in enumerate(cut):
        top[:split, c] = True
        bottom[split:, c] = True
    return [{"side": "up", "mask": top}, {"side": "down", "mask": bottom}]


def _partition_two_vertical(rng: random.Random, h: int, w: int) -> list[dict[str, object]]:
    cut = _profile(rng, h, 2, w - 2)
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    for r, split in enumerate(cut):
        left[r, :split] = True
        right[r, split:] = True
    return [{"side": "left", "mask": left}, {"side": "right", "mask": right}]


def _partition_top_left_right(rng: random.Random, h: int, w: int) -> list[dict[str, object]]:
    top_profile = _profile(rng, w, 2, h - 4)
    split = rng.randint(2, w - 2)
    top = np.zeros((h, w), dtype=bool)
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if r < top_profile[c]:
                top[r, c] = True
            elif c < split:
                left[r, c] = True
            else:
                right[r, c] = True
    return [{"side": "up", "mask": top}, {"side": "left", "mask": left}, {"side": "right", "mask": right}]


def _partition_bottom_left_right(rng: random.Random, h: int, w: int) -> list[dict[str, object]]:
    bottom_profile = _profile(rng, w, 2, h - 4)
    split = rng.randint(2, w - 2)
    bottom = np.zeros((h, w), dtype=bool)
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if r >= h - bottom_profile[c]:
                bottom[r, c] = True
            elif c < split:
                left[r, c] = True
            else:
                right[r, c] = True
    return [{"side": "down", "mask": bottom}, {"side": "left", "mask": left}, {"side": "right", "mask": right}]


def _partition_left_top_bottom(rng: random.Random, h: int, w: int) -> list[dict[str, object]]:
    left_profile = _profile(rng, h, 2, w - 4)
    split = rng.randint(2, h - 2)
    left = np.zeros((h, w), dtype=bool)
    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if c < left_profile[r]:
                left[r, c] = True
            elif r < split:
                top[r, c] = True
            else:
                bottom[r, c] = True
    return [{"side": "left", "mask": left}, {"side": "up", "mask": top}, {"side": "down", "mask": bottom}]


def _partition_right_top_bottom(rng: random.Random, h: int, w: int) -> list[dict[str, object]]:
    right_profile = _profile(rng, h, 2, w - 4)
    split = rng.randint(2, h - 2)
    right = np.zeros((h, w), dtype=bool)
    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if c >= w - right_profile[r]:
                right[r, c] = True
            elif r < split:
                top[r, c] = True
            else:
                bottom[r, c] = True
    return [{"side": "right", "mask": right}, {"side": "up", "mask": top}, {"side": "down", "mask": bottom}]


def _partition_top_bottom_left_right(rng: random.Random, h: int, w: int) -> list[dict[str, object]]:
    top_h = rng.randint(2, h - 4)
    bottom_h = rng.randint(2, h - top_h - 2)
    split = _profile(rng, h - top_h - bottom_h, 2, w - 2)
    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    top[:top_h, :] = True
    bottom[h - bottom_h :, :] = True
    for offset, r in enumerate(range(top_h, h - bottom_h)):
        cut = split[offset]
        left[r, :cut] = True
        right[r, cut:] = True
    return [
        {"side": "up", "mask": top},
        {"side": "down", "mask": bottom},
        {"side": "left", "mask": left},
        {"side": "right", "mask": right},
    ]


def _partition_left_right_top_bottom(rng: random.Random, h: int, w: int) -> list[dict[str, object]]:
    left_w = rng.randint(2, w - 4)
    right_w = rng.randint(2, w - left_w - 2)
    split = _profile(rng, w - left_w - right_w, 2, h - 2)
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    left[:, :left_w] = True
    right[:, w - right_w :] = True
    for offset, c in enumerate(range(left_w, w - right_w)):
        cut = split[offset]
        top[:cut, c] = True
        bottom[cut:, c] = True
    return [
        {"side": "left", "mask": left},
        {"side": "right", "mask": right},
        {"side": "up", "mask": top},
        {"side": "down", "mask": bottom},
    ]


def _build_partition(rng: random.Random) -> tuple[int, int, list[dict[str, object]]] | None:
    count = rng.randint(2, 4)
    h = rng.randint(*RECT_H_RANGE)
    w = rng.randint(*RECT_W_RANGE)

    if count == 2:
        pieces = rng.choice((_partition_two_horizontal, _partition_two_vertical))(rng, h, w)
    elif count == 3:
        pieces = rng.choice(
            (
                _partition_top_left_right,
                _partition_bottom_left_right,
                _partition_left_top_bottom,
                _partition_right_top_bottom,
            )
        )(rng, h, w)
    else:
        pieces = rng.choice((_partition_top_bottom_left_right, _partition_left_right_top_bottom))(rng, h, w)

    total = np.zeros((h, w), dtype=int)
    for piece in pieces:
        total += np.asarray(piece["mask"], dtype=int)
        mask = np.asarray(piece["mask"], dtype=bool)
        if not _is_connected(mask):
            return None
        if not _has_full_2x2(mask):
            return None
        if int(mask.sum()) < 5:
            return None
    if not np.array_equal(total, np.ones((h, w), dtype=int)):
        return None
    return h, w, pieces


def _arrow_slots(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if not bool(mask[r : r + 2, c : c + 2].all()):
                continue
            fallback.append((r, c))
            if 0 < r < h - 2 and 0 < c < w - 2:
                preferred.append((r, c))
    return preferred if preferred else fallback


def _render_piece_patch(rng: random.Random, mask: np.ndarray, color: int, side: str) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    slots = _arrow_slots(mask)
    assert slots, "piece must have a valid 2x2 marker slot"
    r0, c0 = rng.choice(slots)
    for dr, dc in ARROW_PATTERNS[side]:
        patch[r0 + dr, c0 + dc] = GREY
    return patch


def _place_bbox(
    rng: random.Random,
    blocked: np.ndarray,
    box_h: int,
    box_w: int,
) -> tuple[int, int] | None:
    gh, gw = blocked.shape
    if box_h > gh or box_w > gw:
        return None

    min_r = 1 if gh - box_h >= 2 else 0
    max_r = gh - box_h - 1 if gh - box_h >= 2 else gh - box_h
    min_c = 1 if gw - box_w >= 2 else 0
    max_c = gw - box_w - 1 if gw - box_w >= 2 else gw - box_w
    if max_r < min_r or max_c < min_c:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(min_r, max_r)
        c0 = rng.randint(min_c, max_c)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(gh, r0 + box_h + 1)
        cc1 = min(gw, c0 + box_w + 1)
        if bool(blocked[rr0:rr1, cc0:cc1].any()):
            continue
        blocked[rr0:rr1, cc0:cc1] = True
        return int(r0), int(c0)
    return None


def _paste_non_bg(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_RETRIES):
        built = _build_partition(rng)
        if built is None:
            continue
        target_h, target_w, pieces = built
        color = rng.choice(COLOR_CHOICES)
        grid_h = rng.randint(*GRID_H_RANGE)
        grid_w = rng.randint(*GRID_W_RANGE)
        frame_h = target_h + 2
        frame_w = target_w + 2
        if frame_h + 2 > grid_h or frame_w + 2 > grid_w:
            continue

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)

        frame_origin = _place_bbox(rng, blocked, frame_h, frame_w)
        if frame_origin is None:
            continue
        fr, fc = frame_origin
        frame_bbox = (fr, fc, fr + frame_h - 1, fc + frame_w - 1)
        _draw_frame(grid, frame_bbox, GREY)

        rendered: list[np.ndarray] = []
        for piece in pieces:
            rendered.append(_render_piece_patch(rng, np.asarray(piece["mask"], dtype=bool), color, str(piece["side"])))

        ok = True
        for patch in rendered:
            origin = _place_bbox(rng, blocked, patch.shape[0], patch.shape[1])
            if origin is None:
                ok = False
                break
            _paste_non_bg(grid, patch, origin[0], origin[1])
        if not ok:
            continue

        try:
            parsed_frame, parsed_pieces = _extract_scene(grid)
            pr0, pc0, pr1, pc1 = parsed_frame
            sols = _solve_cover(parsed_pieces, pr1 - pr0 - 1, pc1 - pc0 - 1, max_solutions=2)
            if len(sols) != 1:
                continue
            out = main(grid)
            if np.array_equal(out, grid):
                continue
        except Exception:
            continue

        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
