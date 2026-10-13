import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
BLUE = 1
RED = 2
GREEN = 3

EJECT_COLORS = (RED, BLUE, GREEN)
NEUTRAL_COLORS = (4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 500
CASE_RETRIES = 200
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 120
TEMP_CANVAS = 30


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _same_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _interior_components(grid: np.ndarray, bbox_: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox_
    sub = grid[r0 + 1 : r1, c0 + 1 : c1]
    sh, sw = sub.shape
    visited = np.zeros((sh, sw), dtype=bool)
    pieces = []
    for r in range(sh):
        for c in range(sw):
            color = int(sub[r, c])
            if visited[r, c] or color in (BG, FRAME):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < sh and 0 <= nc < sw and not visited[nr, nc] and int(sub[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            local = np.array(cells, dtype=int)
            lr0, lc0, lr1, lc1 = _bbox(local)
            patch = np.zeros((lr1 - lr0 + 1, lc1 - lc0 + 1), dtype=int)
            patch[local[:, 0] - lr0, local[:, 1] - lc0] = color
            abs_coords = local + np.array([[r0 + 1, c0 + 1]])
            pieces.append(
                {
                    "color": color,
                    "coords": abs_coords,
                    "bbox": (int(lr0 + r0 + 1), int(lc0 + c0 + 1), int(lr1 + r0 + 1), int(lc1 + c0 + 1)),
                    "patch": patch,
                    "height": int(patch.shape[0]),
                    "width": int(patch.shape[1]),
                }
            )
    return pieces


def _transform_piece(patch: np.ndarray, color: int, side: str) -> np.ndarray:
    if int(color) == RED:
        return patch.copy()
    if int(color) == BLUE:
        return np.flipud(patch) if side in ("top", "bottom") else np.fliplr(patch)
    if int(color) == GREEN:
        return np.rot90(patch, 2)
    raise AssertionError(f"unsupported eject color: {color}")


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    h, w = dst.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        raise AssertionError("patch placement exceeds grid bounds")
    for rr, cc in np.argwhere(patch != BG):
        tr = int(r0 + rr)
        tc = int(c0 + cc)
        if int(dst[tr, tc]) != BG:
            raise AssertionError("patch placement overlaps existing content")
        dst[tr, tc] = int(patch[rr, cc])


def _frame_gap(mask: np.ndarray) -> tuple[str, int, int]:
    sides = []
    side_arrays = {
        "top": mask[0, :],
        "bottom": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }
    for side, arr in side_arrays.items():
        missing = np.flatnonzero(~arr)
        if missing.size == 0:
            continue
        if missing[0] == 0 or missing[-1] == arr.size - 1:
            raise AssertionError("gap must stay away from frame corners")
        if missing.size != (missing[-1] - missing[0] + 1):
            raise AssertionError("gap must be contiguous")
        if arr[: missing[0]].all() and arr[missing[-1] + 1 :].all():
            sides.append((side, int(missing[0]), int(missing.size)))
    if len(sides) != 1:
        raise AssertionError("expected exactly one frame gap")
    return sides[0]


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _same_color_components(grid, FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if bool(mask[1:-1, 1:-1].any()):
            continue
        try:
            side, offset, gap_len = _frame_gap(mask)
        except AssertionError:
            continue
        expected_area = 2 * mask.shape[0] + 2 * mask.shape[1] - 4 - gap_len
        if int(mask.sum()) != int(expected_area):
            continue
        if side in ("top", "bottom"):
            gap_start = c0 + offset
        else:
            gap_start = r0 + offset
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "side": side,
                "gap_start": int(gap_start),
                "gap_len": int(gap_len),
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject the unique gap-matching fragment out of each holed frame and transform it by color.

    Input structure:
    - Background is black.
    - Each gray 1-pixel-thick hollow rectangle has exactly one contiguous gap on one side.
    - Each frame contains one or more monochrome 4-connected fragments, and exactly one fragment has the same side-length as the gap along the gap's orientation.

    Rule:
    - Detect every gray frame, its gap side, and the gap length.
    - Inside each frame, select the unique fragment whose bbox width matches a top/bottom gap or whose bbox height matches a left/right gap.
    - Remove that fragment from the frame interior and place it immediately outside the frame, aligned to the gap span and adjacent to the outer wall.
    - Red fragments keep their shape, blue fragments mirror along the ejection direction, and green fragments rotate 180 degrees.

    Key insight:
    The gap determines both which fragment moves and where it exits, while the moved fragment's visible color independently encodes the post-ejection shape transform.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    if not frames:
        raise AssertionError("no valid gray frames found")

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        side = frame["side"]
        gap_len = int(frame["gap_len"])
        pieces = _interior_components(g, frame["bbox"])
        candidates = [
            piece
            for piece in pieces
            if (piece["width"] == gap_len if side in ("top", "bottom") else piece["height"] == gap_len)
        ]
        if len(candidates) != 1:
            raise AssertionError("frame must contain exactly one ejectable fragment")
        piece = candidates[0]
        transformed = _transform_piece(piece["patch"], int(piece["color"]), side)

        for pr, pc in piece["coords"]:
            out[int(pr), int(pc)] = BG

        if side == "top":
            target_r0 = r0 - piece["height"]
            target_c0 = int(frame["gap_start"])
        elif side == "bottom":
            target_r0 = r1 + 1
            target_c0 = int(frame["gap_start"])
        elif side == "left":
            target_r0 = int(frame["gap_start"])
            target_c0 = c0 - piece["width"]
        else:
            target_r0 = int(frame["gap_start"])
            target_c0 = c1 + 1

        _paste_non_bg(out, transformed, int(target_r0), int(target_c0))

    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _shape_asymmetric(mask: np.ndarray, mode: str | None) -> bool:
    if mode is None:
        return True
    if mode == "flipud":
        return not np.array_equal(mask, np.flipud(mask))
    if mode == "fliplr":
        return not np.array_equal(mask, np.fliplr(mask))
    if mode == "rot180":
        return not np.array_equal(mask, np.rot90(mask, 2))
    raise AssertionError(f"unknown asymmetry mode: {mode}")


def _random_shape(h: int, w: int, asymmetry: str | None = None) -> np.ndarray | None:
    min_area = max(1, h + w - 1) if h > 1 and w > 1 else max(h, w)
    max_area = h * w
    if asymmetry is not None and max_area > min_area:
        max_area -= 1
    for _ in range(SHAPE_RETRIES):
        target_area = random.randint(min_area, max_area)
        mask = np.ones((h, w), dtype=bool)
        while int(mask.sum()) > target_area:
            removable = []
            for r in range(h):
                for c in range(w):
                    if not mask[r, c]:
                        continue
                    mask[r, c] = False
                    if _touches_bbox(mask) and _connected(mask):
                        removable.append((r, c))
                    mask[r, c] = True
            if not removable:
                break
            rr, cc = random.choice(removable)
            mask[rr, cc] = False
        if int(mask.sum()) != target_area:
            continue
        if not _touches_bbox(mask):
            continue
        if not _shape_asymmetric(mask, asymmetry):
            continue
        return mask
    return None


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    ih, iw = occupied.shape
    for rr, cc in np.argwhere(mask):
        ar = int(r0 + rr)
        ac = int(c0 + cc)
        if occupied[ar, ac]:
            return False
        for nr, nc in _neighbors4(ar, ac):
            if 0 <= nr < ih and 0 <= nc < iw and occupied[nr, nc]:
                return False
    return True


def _paint(inner: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        inner[int(r0 + rr), int(c0 + cc)] = int(color)
        occupied[int(r0 + rr), int(c0 + cc)] = True
    return coords + np.array([[r0, c0]])


def _expected_target(frame_r0: int, frame_c0: int, frame_h: int, frame_w: int, piece_h: int, piece_w: int, side: str, inner_r0: int, inner_c0: int) -> tuple[int, int]:
    if side == "top":
        return frame_r0 - piece_h, frame_c0 + 1 + inner_c0
    if side == "bottom":
        return frame_r0 + frame_h, frame_c0 + 1 + inner_c0
    if side == "left":
        return frame_r0 + 1 + inner_r0, frame_c0 - piece_w
    return frame_r0 + 1 + inner_r0, frame_c0 + frame_w


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(CASE_RETRIES):
        side = random.choice(SIDES)
        color = random.choice(EJECT_COLORS)
        inner_h = random.randint(4, 6)
        inner_w = random.randint(4, 6)
        piece_h = random.randint(2, min(4, inner_h))
        piece_w = random.randint(2, min(4, inner_w))

        asymmetry = None
        if color == BLUE:
            asymmetry = "flipud" if side in ("top", "bottom") else "fliplr"
        elif color == GREEN:
            asymmetry = "rot180"

        piece_mask = _random_shape(piece_h, piece_w, asymmetry=asymmetry)
        if piece_mask is None:
            continue

        gap_len = piece_w if side in ("top", "bottom") else piece_h
        piece_count = random.randint(1, 3)
        inner = np.zeros((inner_h, inner_w), dtype=int)
        occupied = np.zeros((inner_h, inner_w), dtype=bool)

        placed = False
        for _place_try in range(PLACEMENT_RETRIES):
            pr0 = random.randint(0, inner_h - piece_h)
            pc0 = random.randint(0, inner_w - piece_w)
            if not _can_place(occupied, piece_mask, pr0, pc0):
                continue
            piece_coords = _paint(inner, occupied, piece_mask, color, pr0, pc0)
            placed = True
            break
        if not placed:
            continue

        ok = True
        neutral_pool = list(NEUTRAL_COLORS)
        random.shuffle(neutral_pool)
        for idx in range(piece_count - 1):
            if idx >= len(neutral_pool):
                ok = False
                break
            distractor_placed = False
            for _dist_try in range(PLACEMENT_RETRIES):
                dh = random.randint(1, min(3, inner_h))
                dw = random.randint(1, min(3, inner_w))
                if dh == gap_len or dw == gap_len:
                    continue
                dist_mask = _random_shape(dh, dw)
                if dist_mask is None:
                    continue
                dr0 = random.randint(0, inner_h - dh)
                dc0 = random.randint(0, inner_w - dw)
                if not _can_place(occupied, dist_mask, dr0, dc0):
                    continue
                _paint(inner, occupied, dist_mask, neutral_pool[idx], dr0, dc0)
                distractor_placed = True
                break
            if not distractor_placed:
                ok = False
                break
        if not ok:
            continue

        frame_h = inner_h + 2
        frame_w = inner_w + 2
        top_pad = piece_h if side == "top" else 1
        bottom_pad = piece_h if side == "bottom" else 1
        left_pad = piece_w if side == "left" else 1
        right_pad = piece_w if side == "right" else 1

        patch_h = top_pad + frame_h + bottom_pad
        patch_w = left_pad + frame_w + right_pad
        if patch_h > TEMP_CANVAS or patch_w > TEMP_CANVAS:
            continue

        frame_r0 = top_pad
        frame_c0 = left_pad
        frame_r1 = frame_r0 + frame_h - 1
        frame_c1 = frame_c0 + frame_w - 1

        inp = np.zeros((patch_h, patch_w), dtype=int)
        inp[frame_r0, frame_c0 : frame_c1 + 1] = FRAME
        inp[frame_r1, frame_c0 : frame_c1 + 1] = FRAME
        inp[frame_r0 : frame_r1 + 1, frame_c0] = FRAME
        inp[frame_r0 : frame_r1 + 1, frame_c1] = FRAME

        gap_r0 = frame_r0 + 1 + pr0
        gap_c0 = frame_c0 + 1 + pc0
        if side == "top":
            inp[frame_r0, gap_c0 : gap_c0 + piece_w] = BG
        elif side == "bottom":
            inp[frame_r1, gap_c0 : gap_c0 + piece_w] = BG
        elif side == "left":
            inp[gap_r0 : gap_r0 + piece_h, frame_c0] = BG
        else:
            inp[gap_r0 : gap_r0 + piece_h, frame_c1] = BG

        inp[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1] = inner
        out = inp.copy()

        for rr, cc in piece_coords:
            out[int(frame_r0 + 1 + rr), int(frame_c0 + 1 + cc)] = BG

        piece_patch = np.zeros((piece_h, piece_w), dtype=int)
        piece_patch[piece_mask] = color
        moved = _transform_piece(piece_patch, color, side)
        target_r0, target_c0 = _expected_target(frame_r0, frame_c0, frame_h, frame_w, piece_h, piece_w, side, pr0, pc0)
        try:
            _paste_non_bg(out, moved, int(target_r0), int(target_c0))
        except AssertionError:
            continue

        return inp, out
    return None


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    nr0 = r0 - pad
    nc0 = c0 - pad
    nr1 = r0 + h - 1 + pad
    nc1 = c0 + w - 1 + pad
    for ar0, ac0, ar1, ac1 in rects:
        if not (nr1 < ar0 or ar1 < nr0 or nc1 < ac0 or ac1 < nc0):
            return False
    return True


def _compose_cases(cases: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((TEMP_CANVAS, TEMP_CANVAS), dtype=int)
    out = np.zeros_like(inp)
    rects: list[tuple[int, int, int, int]] = []
    shuffled = list(cases)
    random.shuffle(shuffled)
    for in_patch, out_patch in shuffled:
        ph, pw = in_patch.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES * 2):
            r0 = random.randint(0, TEMP_CANVAS - ph)
            c0 = random.randint(0, TEMP_CANVAS - pw)
            if not _rect_separated(rects, r0, c0, ph, pw, pad=1):
                continue
            inp[r0 : r0 + ph, c0 : c0 + pw] = in_patch
            out[r0 : r0 + ph, c0 : c0 + pw] = out_patch
            rects.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
            placed = True
            break
        if not placed:
            return None

    min_r = min(r0 for r0, _c0, _r1, _c1 in rects)
    min_c = min(c0 for _r0, c0, _r1, _c1 in rects)
    max_r = max(r1 for _r0, _c0, r1, _c1 in rects)
    max_c = max(c1 for _r0, _c0, _r1, c1 in rects)
    inp = inp[min_r : max_r + 1, min_c : max_c + 1]
    out = out[min_r : max_r + 1, min_c : max_c + 1]

    free_h = TEMP_CANVAS - inp.shape[0]
    free_w = TEMP_CANVAS - inp.shape[1]
    top = random.randint(0, min(4, free_h))
    bottom = random.randint(0, min(4, free_h - top))
    left = random.randint(0, min(4, free_w))
    right = random.randint(0, min(4, free_w - left))
    inp = np.pad(inp, ((top, bottom), (left, right)), constant_values=BG)
    out = np.pad(out, ((top, bottom), (left, right)), constant_values=BG)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_cases = random.randint(1, 3)
    cases = []
    for _ in range(n_cases):
        case = _build_case()
        if case is None:
            return None
        cases.append(case)

    composed = _compose_cases(cases)
    if composed is None:
        return None
    inp, expected = composed
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    counts = np.bincount(inp.ravel(), minlength=10)
    if int(counts[BG]) < int(counts.max()):
        return None
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
