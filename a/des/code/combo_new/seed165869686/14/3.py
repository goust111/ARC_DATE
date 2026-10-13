from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_SIDES = ("top", "bottom", "left", "right")
_BG = 0


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
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
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out, r0, c0


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _transform_by_side(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return np.flipud(mask)
    if side == "left":
        return np.fliplr(mask)
    return mask.copy()


def _piece_side(bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if r1 < fr0:
        return "top"
    if r0 > fr1:
        return "bottom"
    if c1 < fc0:
        return "left"
    if c0 > fc1:
        return "right"
    raise AssertionError("outside pieces must lie wholly on one side of the frame")


def _extract_notches(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame_bbox
    sub = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
    notches = []
    for comp in _components(sub == bg):
        sides = []
        if np.any(comp[:, 0] == 0):
            sides.append("top")
        if np.any(comp[:, 0] == sub.shape[0] - 1):
            sides.append("bottom")
        if np.any(comp[:, 1] == 0):
            sides.append("left")
        if np.any(comp[:, 1] == sub.shape[1] - 1):
            sides.append("right")
        assert len(sides) == 1, "each cavity must be a single-side notch"
        mask, r0, c0 = _tight_mask(comp)
        notches.append(
            {
                "side": sides[0],
                "mask": mask,
                "r0": fr0 + r0,
                "c0": fc0 + c0,
                "used": False,
            }
        )
    assert notches, "expected at least one notch in the frame"
    return notches


def _extract_pieces(grid: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    pieces = []
    for comp in _components((grid != bg) & (grid != frame_color)):
        mask, r0, c0 = _tight_mask(comp)
        colors = {int(grid[int(r), int(c)]) for r, c in comp}
        assert len(colors) == 1, "each fragment should be monochrome"
        color = int(next(iter(colors)))
        bbox = (int(comp[:, 0].min()), int(comp[:, 1].min()), int(comp[:, 0].max()), int(comp[:, 1].max()))
        pieces.append(
            {
                "color": color,
                "mask": mask,
                "coords": comp,
                "side": _piece_side(bbox, frame_bbox),
            }
        )
    assert pieces, "expected scattered fragments outside the frame"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered side fragments back into the matching notches of a central frame.

    Input structure:
    - The background is the dominant color. Near the center sits one large monochrome rectangular frame body whose boundary has several background notches cut into it.
    - Outside the frame are smaller connected monochrome fragments in other colors, each lying wholly above, below, left, or right of the frame.
    - Every notch is a 4-connected background cavity inside the frame bounding box that touches exactly one side of that box.

    Rule:
    - Identify the frame as the largest non-background color region and read its bounding box as the intended full rectangle.
    - Extract each notch inside that box and each outside fragment. A fragment from above is flipped vertically, a fragment from the left is flipped horizontally, and fragments from below or right are kept as-is.
    - Match every transformed fragment to the unique notch on the same side whose shape is identical, then place the fragment's color into that notch.
    - Clear the fragment's original outside location to background and keep all remaining frame pixels unchanged.

    Key insight:
    - Shape chooses the destination notch, but the needed reflection is not a property of the fragment itself; it is determined only by which side of the frame the fragment starts from.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    non_bg = g[g != bg]
    assert non_bg.size > 0, "input must contain a frame"
    frame_color = int(Counter(map(int, non_bg.tolist())).most_common(1)[0][0])
    frame_bbox = _bbox_from_mask(g == frame_color)
    assert frame_bbox is not None, "frame color must form a visible rectangle"

    notches = _extract_notches(g, bg, frame_bbox)
    pieces = _extract_pieces(g, bg, frame_color, frame_bbox)

    out = g.copy()
    for piece in pieces:
        transformed = _transform_by_side(piece["mask"], piece["side"])
        candidates = [
            notch
            for notch in notches
            if not notch["used"] and notch["side"] == piece["side"] and np.array_equal(notch["mask"], transformed)
        ]
        assert len(candidates) == 1, "each fragment must match exactly one notch on its side"
        notch = candidates[0]
        notch["used"] = True

        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg
        nr0 = int(notch["r0"])
        nc0 = int(notch["c0"])
        nh, nw = notch["mask"].shape
        region = out[nr0 : nr0 + nh, nc0 : nc0 + nw]
        region[notch["mask"]] = int(piece["color"])

    assert all(notch["used"] for notch in notches), "every notch should receive exactly one piece"
    return out


def _sample_intervals(total_len: int, count: int, min_len: int, max_len: int, margin: int, gap: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        spans: list[tuple[int, int]] = []
        ok = True
        for _idx in range(count):
            length = random.randint(min_len, max_len)
            starts = list(range(margin, total_len - margin - length + 1))
            random.shuffle(starts)
            placed = False
            for start in starts:
                end = start + length
                if all(end + gap <= s or e + gap <= start for s, e in spans):
                    spans.append((start, end))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            spans.sort()
            return spans
    return None


def _sample_profile(length: int, max_depth: int) -> list[int]:
    for _ in range(100):
        depths = [random.randint(1, max_depth) for _ in range(length)]
        if max(depths) < 2:
            continue
        if sum(depths) < 3:
            continue
        return depths
    raise AssertionError("failed to sample a non-trivial profile")


def _profile_mask(side: str, span_len: int, max_depth: int) -> np.ndarray:
    depths = _sample_profile(span_len, max_depth)
    if side in ("top", "bottom"):
        mask = np.zeros((max_depth, span_len), dtype=bool)
        for c, depth in enumerate(depths):
            if side == "top":
                mask[:depth, c] = True
            else:
                mask[max_depth - depth :, c] = True
        return mask

    mask = np.zeros((span_len, max_depth), dtype=bool)
    for r, depth in enumerate(depths):
        if side == "left":
            mask[r, :depth] = True
        else:
            mask[r, max_depth - depth :] = True
    return mask


def _can_place(mask: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = patch.shape
    H, W = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for pr, pc in np.argwhere(patch):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        r1 = max(0, rr - pad)
        r2 = min(H, rr + pad + 1)
        c1 = max(0, cc - pad)
        c2 = min(W, cc + pad + 1)
        if np.any(mask[r1:r2, c1:c2]):
            return False
    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    region = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    region[mask] = int(color)


def _notch_origin(side: str, frame_top: int, frame_left: int, frame_h: int, frame_w: int, start: int, mask: np.ndarray) -> tuple[int, int]:
    if side == "top":
        return frame_top, frame_left + start
    if side == "bottom":
        return frame_top + frame_h - mask.shape[0], frame_left + start
    if side == "left":
        return frame_top + start, frame_left
    return frame_top + start, frame_left + frame_w - mask.shape[1]


def _build_notches(frame_h: int, frame_w: int, side_counts: dict[str, int]) -> list[dict] | None:
    margin = 4
    gap = 1
    all_notches: list[dict] = []
    seen_keys: set[tuple[str, tuple[tuple[int, int], bytes]]] = set()

    for side in _SIDES:
        count = int(side_counts[side])
        axis_len = frame_w if side in ("top", "bottom") else frame_h
        max_span = 5 if count == 1 else 3
        spans = _sample_intervals(axis_len, count, 2, max_span, margin, gap)
        if spans is None:
            return None

        for start, end in spans:
            span_len = int(end - start)
            for _ in range(100):
                depth = random.randint(2, 4)
                mask = _profile_mask(side, span_len, depth)
                key = (side, _shape_key(mask))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_notches.append({"side": side, "start": int(start), "mask": mask})
                break
            else:
                return None
    return all_notches


def _side_counts() -> dict[str, int]:
    counts = {side: 1 for side in _SIDES}
    extra = random.choice([1, 2])
    for side in random.sample(list(_SIDES), extra):
        counts[side] += 1
    return counts


def _side_positions(
    side: str,
    patch: np.ndarray,
    grid_h: int,
    grid_w: int,
    frame_top: int,
    frame_left: int,
    frame_h: int,
    frame_w: int,
) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    if side == "top":
        row_max = frame_top - ph - 1
        for r in range(0, row_max + 1):
            for c in range(frame_left, frame_left + frame_w - pw + 1):
                positions.append((r, c))
    elif side == "bottom":
        row_min = frame_top + frame_h + 1
        for r in range(row_min, grid_h - ph + 1):
            for c in range(frame_left, frame_left + frame_w - pw + 1):
                positions.append((r, c))
    elif side == "left":
        col_max = frame_left - pw - 1
        for r in range(frame_top, frame_top + frame_h - ph + 1):
            for c in range(0, col_max + 1):
                positions.append((r, c))
    else:
        col_min = frame_left + frame_w + 1
        for r in range(frame_top, frame_top + frame_h - ph + 1):
            for c in range(col_min, grid_w - pw + 1):
                positions.append((r, c))
    random.shuffle(positions)
    return positions


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    frame_h = random.randint(15, 18)
    frame_w = random.randint(15, 18)
    top_margin = random.randint(5, 6)
    bottom_margin = random.randint(5, 6)
    left_margin = random.randint(5, 6)
    right_margin = random.randint(5, 6)
    grid_h = frame_h + top_margin + bottom_margin
    grid_w = frame_w + left_margin + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    side_counts = _side_counts()
    notches = _build_notches(frame_h, frame_w, side_counts)
    if notches is None:
        return None

    colors = list(range(1, 10))
    random.shuffle(colors)
    frame_color = colors.pop()
    if len(colors) < len(notches):
        return None
    piece_colors = colors[: len(notches)]

    frame_top = top_margin
    frame_left = left_margin

    inp = np.full((grid_h, grid_w), _BG, dtype=int)
    out = np.full((grid_h, grid_w), _BG, dtype=int)
    inp[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame_color
    out[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame_color

    piece_specs: list[dict] = []
    for notch, color in zip(notches, piece_colors, strict=True):
        side = str(notch["side"])
        mask = np.asarray(notch["mask"], dtype=bool)
        nr0, nc0 = _notch_origin(side, frame_top, frame_left, frame_h, frame_w, int(notch["start"]), mask)
        _paint_mask(inp, mask, nr0, nc0, _BG)
        _paint_mask(out, mask, nr0, nc0, int(color))
        piece_specs.append({"side": side, "mask": mask, "color": int(color)})

    occupied = inp != _BG
    for spec in piece_specs:
        piece_mask = _transform_by_side(spec["mask"], spec["side"])
        positions = _side_positions(
            spec["side"],
            piece_mask,
            grid_h,
            grid_w,
            frame_top,
            frame_left,
            frame_h,
            frame_w,
        )
        placed = False
        for r0, c0 in positions:
            if _can_place(occupied, piece_mask, int(r0), int(c0), pad=1):
                _paint_mask(inp, piece_mask, int(r0), int(c0), int(spec["color"]))
                occupied[r0 : r0 + piece_mask.shape[0], c0 : c0 + piece_mask.shape[1]] |= piece_mask
                placed = True
                break
        if not placed:
            return None

    try:
        if np.array_equal(main(inp), out):
            return inp, out
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid ARC example")
