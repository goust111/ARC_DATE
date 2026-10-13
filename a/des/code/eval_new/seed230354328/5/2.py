from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int):
    yield r + 1, c
    yield r - 1, c
    yield r, c + 1
    yield r, c - 1


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
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _find_frame(grid: np.ndarray, bg: int):
    for comp in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) == expected:
            return {
                "color": int(grid[comp[0][0], comp[0][1]]),
                "bbox": (r0, c0, r1, c1),
                "cells": expected,
            }
    raise AssertionError("frame not found")


_MISSING_TO_QUAD = {
    (0, 0): "BR",
    (0, 1): "BL",
    (1, 0): "TR",
    (1, 1): "TL",
}


def _piece_info(grid: np.ndarray, comp: list[tuple[int, int]]):
    counts = Counter(int(grid[r, c]) for r, c in comp)
    piece_color = counts.most_common(1)[0][0]
    arrow_colors = [color for color in counts if color != piece_color]
    assert len(arrow_colors) == 1, "each piece should have exactly one arrow color"
    arrow_color = arrow_colors[0]
    arrow_cells = [(r, c) for r, c in comp if int(grid[r, c]) == arrow_color]
    assert len(arrow_cells) == 3, "arrow marker must use exactly 3 cells"
    ar0, ac0, ar1, ac1 = _bbox(arrow_cells)
    assert ar1 - ar0 == 1 and ac1 - ac0 == 1, "arrow marker must occupy a 2x2 box"
    local_arrow = {(r - ar0, c - ac0) for r, c in arrow_cells}
    missing = next(corner for corner in ((0, 0), (0, 1), (1, 0), (1, 1)) if corner not in local_arrow)
    target_quad = _MISSING_TO_QUAD[missing]

    r0, c0, r1, c1 = _bbox(comp)
    local_cells = {(r - r0, c - c0) for r, c in comp}
    return {
        "quad": target_quad,
        "piece_color": piece_color,
        "bbox_shape": (r1 - r0 + 1, c1 - c0 + 1),
        "cells": local_cells,
        "centroid": (
            sum(r for r, _ in local_cells) / len(local_cells),
            sum(c for _, c in local_cells) / len(local_cells),
        ),
    }


def _corner_translation(quad: str, piece_h: int, piece_w: int, frame_h: int, frame_w: int) -> tuple[int, int]:
    if quad == "TL":
        return 0, 0
    if quad == "TR":
        return 0, frame_w - piece_w
    if quad == "BL":
        return frame_h - piece_h, 0
    if quad == "BR":
        return frame_h - piece_h, frame_w - piece_w
    raise AssertionError(f"unknown quadrant: {quad}")


def _quadrant_ok(quad: str, centroid_r: float, centroid_c: float, frame_h: int, frame_w: int) -> bool:
    top = centroid_r < frame_h / 2.0
    left = centroid_c < frame_w / 2.0
    here = ("T" if top else "B") + ("L" if left else "R")
    return here == quad


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _full_span(cells: set[tuple[int, int]], frame_h: int, frame_w: int) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return r0 == 0 and c0 == 0 and r1 == frame_h - 1 and c1 == frame_w - 1


def _contact_score(placed: list[dict], frame_h: int, frame_w: int) -> int:
    owner: dict[tuple[int, int], int] = {}
    for idx, item in enumerate(placed):
        for cell in item["translated"]:
            owner[cell] = idx

    score = 0
    for (r, c), idx in owner.items():
        for nr, nc in ((r + 1, c), (r, c + 1)):
            other = owner.get((nr, nc))
            if other is not None and other != idx:
                score += 1

    border_touches = 0
    for r, c in owner:
        if r in (0, frame_h - 1):
            border_touches += 1
        if c in (0, frame_w - 1):
            border_touches += 1
    return score * 10 + border_touches


def _assemble_frame(pieces: list[dict], frame_h: int, frame_w: int) -> set[tuple[int, int]]:
    corner_placed: list[dict] = []
    for piece in pieces:
        ph, pw = piece["bbox_shape"]
        dr, dc = _corner_translation(piece["quad"], ph, pw, frame_h, frame_w)
        translated = _translate(piece["cells"], dr, dc)
        if any(r < 0 or c < 0 or r >= frame_h or c >= frame_w for r, c in translated):
            break
        corner_placed.append({"piece": piece, "translated": translated})
    else:
        union: set[tuple[int, int]] = set()
        ok = True
        for item in corner_placed:
            if union & item["translated"]:
                ok = False
                break
            union |= item["translated"]
        if ok and _full_span(union, frame_h, frame_w) and _is_connected(union):
            return union

    candidates: list[list[dict]] = []
    for piece in pieces:
        ph, pw = piece["bbox_shape"]
        options: list[dict] = []
        for dr in range(frame_h - ph + 1):
            for dc in range(frame_w - pw + 1):
                centroid_r = piece["centroid"][0] + dr
                centroid_c = piece["centroid"][1] + dc
                if not _quadrant_ok(piece["quad"], centroid_r, centroid_c, frame_h, frame_w):
                    continue
                translated = _translate(piece["cells"], dr, dc)
                options.append({"dr": dr, "dc": dc, "translated": translated})
        assert options, "piece has no candidate placements"
        candidates.append(options)

    order = sorted(range(len(pieces)), key=lambda idx: len(candidates[idx]))
    best_union: set[tuple[int, int]] | None = None
    best_score = -1
    placements: list[dict | None] = [None] * len(pieces)

    def search(depth: int, occupied: set[tuple[int, int]]):
        nonlocal best_union, best_score
        if depth == len(order):
            if not occupied or not _full_span(occupied, frame_h, frame_w) or not _is_connected(occupied):
                return
            placed = [item for item in placements if item is not None]
            score = _contact_score(placed, frame_h, frame_w)
            if score > best_score:
                best_score = score
                best_union = set(occupied)
            return

        idx = order[depth]
        for option in candidates[idx]:
            translated = option["translated"]
            if occupied & translated:
                continue
            placements[idx] = {"piece": pieces[idx], "translated": translated}
            search(depth + 1, occupied | translated)
            placements[idx] = None

    search(0, set())
    if best_union is None:
        raise AssertionError("failed to assemble pieces inside the frame")
    return best_union


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered marked fragments into the original rectangular picture inside the frame.

    Input structure:
    - The background is black, and one separate rectangular border frame encloses an empty black interior.
    - Four irregular fragments of one shape color are scattered elsewhere; each fragment contains a 3-cell L-shaped marker of another color inside a 2x2 box.
    - The missing corner of that 2x2 marker points diagonally and encodes which opposite output quadrant the fragment belongs to.

    Rule:
    - Identify the frame and the four scattered fragments, then treat every marker cell as ordinary fragment material for the reconstruction.
    - Assemble the fragments by translation only so they fit back into the frame's interior as one rectangular picture; the marker's missing corner maps to the opposite quadrant: bottom-right -> top-left, bottom-left -> top-right, top-right -> bottom-left, top-left -> bottom-right.
    - Draw the reassembled picture inside the frame, remove the scattered originals, and recolor all former marker cells to the fragment color.

    Key insight:
    - The fragment shapes provide the physical jigsaw fit, while the marker's missing corner gives the fragment's opposite output quadrant.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame = _find_frame(g, bg)

    fragment_mask = (g != bg).copy()
    for r, c in frame["cells"]:
        fragment_mask[r, c] = False
    fragments = _components(fragment_mask)
    assert len(fragments) == 4, "expected exactly four fragments"

    pieces = [_piece_info(g, comp) for comp in fragments]
    piece_color = Counter(piece["piece_color"] for piece in pieces).most_common(1)[0][0]

    r0, c0, r1, c1 = frame["bbox"]
    frame_h = r1 - r0 - 1
    frame_w = c1 - c0 - 1
    assembled = _assemble_frame(pieces, frame_h, frame_w)

    out = np.full_like(g, bg)
    for rr in range(r0, r1 + 1):
        out[rr, c0] = frame["color"]
        out[rr, c1] = frame["color"]
    for cc in range(c0, c1 + 1):
        out[r0, cc] = frame["color"]
        out[r1, cc] = frame["color"]
    for rr, cc in assembled:
        out[r0 + 1 + rr, c0 + 1 + cc] = piece_color
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    return _is_connected(cells)


def _random_cut(length: int, start: int, low: int, high: int, rng: random.Random) -> list[int]:
    values = [start]
    for _ in range(length - 1):
        step = rng.choice([-1, 0, 1])
        values.append(max(low, min(high, values[-1] + step)))
    return values


def _choose_arrow_block(mask: np.ndarray) -> tuple[int, int] | None:
    h, w = mask.shape
    strict: list[tuple[int, int]] = []
    relaxed: list[tuple[int, int]] = []
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            if not mask[r : r + 2, c : c + 2].all():
                continue
            relaxed.append((r, c))
            ring = [
                (r - 1, c),
                (r - 1, c + 1),
                (r + 2, c),
                (r + 2, c + 1),
                (r, c - 1),
                (r + 1, c - 1),
                (r, c + 2),
                (r + 1, c + 2),
            ]
            if all(mask[rr, cc] for rr, cc in ring):
                strict.append((r, c))
    if strict:
        return strict[0]
    if relaxed:
        return relaxed[0]
    return None


_QUADS = ("TL", "TR", "BL", "BR")
_QUAD_TO_MISSING = {
    "TL": (1, 1),
    "TR": (1, 0),
    "BL": (0, 1),
    "BR": (0, 0),
}
_QUAD_TO_CORNER = {
    "TL": (0, 0),
    "TR": (0, -1),
    "BL": (-1, 0),
    "BR": (-1, -1),
}


def _build_partition(mask: np.ndarray, rng: random.Random):
    h, w = mask.shape
    for _ in range(80):
        vsep = _random_cut(h, rng.randint(max(2, w // 2 - 1), min(w - 2, w // 2 + 1)), 2, w - 2, rng)
        hsep = _random_cut(w, rng.randint(max(2, h // 2 - 1), min(h - 2, h // 2 + 1)), 2, h - 2, rng)
        piece_masks = {quad: np.zeros((h, w), dtype=bool) for quad in _QUADS}
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                top = r < hsep[c]
                left = c < vsep[r]
                quad = ("T" if top else "B") + ("L" if left else "R")
                piece_masks[quad][r, c] = True

        ok = True
        arrow_blocks: dict[str, tuple[int, int]] = {}
        for quad in _QUADS:
            pmask = piece_masks[quad]
            if pmask.sum() < 8:
                ok = False
                break
            corner_r, corner_c = _QUAD_TO_CORNER[quad]
            if not pmask[corner_r, corner_c]:
                ok = False
                break
            if not _mask_connected(pmask):
                ok = False
                break
            block = _choose_arrow_block(pmask)
            if block is None:
                ok = False
                break
            ys, xs = np.where(pmask)
            centroid_r = float(np.mean(ys))
            centroid_c = float(np.mean(xs))
            if not _quadrant_ok(quad, centroid_r, centroid_c, h, w):
                ok = False
                break
            arrow_blocks[quad] = block
        if ok:
            return piece_masks, arrow_blocks
    return None


def _make_base_mask(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    mask = np.ones((h, w), dtype=bool)
    hole_count = rng.choice([0, 1, 2])
    for _ in range(hole_count):
        placed = False
        for _ in range(20):
            hh = rng.randint(1, min(3, h - 4))
            ww = rng.randint(1, min(3, w - 4))
            r0 = rng.randint(1, h - hh - 1)
            c0 = rng.randint(1, w - ww - 1)
            trial = mask.copy()
            trial[r0 : r0 + hh, c0 : c0 + ww] = False
            if trial[0, 0] and trial[0, -1] and trial[-1, 0] and trial[-1, -1] and _mask_connected(trial):
                mask = trial
                placed = True
                break
        if not placed:
            return None
    return mask


def _reserve(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    reserved[r0:r1, c0:c1] = True


def _can_place_rect(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = reserved.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not reserved[r0:r1, c0:c1].any()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_color, arrow_color, frame_color = rng.sample(list(range(1, 10)), 3)
    inner_h = rng.randint(8, 11)
    inner_w = rng.randint(8, 11)

    mask = _make_base_mask(inner_h, inner_w, rng)
    if mask is None:
        return None
    built = _build_partition(mask, rng)
    if built is None:
        return None
    piece_masks, arrow_blocks = built

    piece_arrays: dict[str, np.ndarray] = {}
    for quad in _QUADS:
        pmask = piece_masks[quad]
        cells = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
        r0, c0, r1, c1 = _bbox(cells)
        sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in cells:
            sprite[r - r0, c - c0] = piece_color
        ar, ac = arrow_blocks[quad]
        miss = _QUAD_TO_MISSING[quad]
        for dr in range(2):
            for dc in range(2):
                if (dr, dc) != miss:
                    sprite[ar - r0 + dr, ac - c0 + dc] = arrow_color
        piece_arrays[quad] = sprite

    canvas_h = rng.randint(max(inner_h + 10, 22), 30)
    canvas_w = rng.randint(max(inner_w + 10, 22), 30)
    if canvas_h > 30 or canvas_w > 30:
        return None

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    reserved = np.zeros((canvas_h, canvas_w), dtype=bool)

    objects = [("frame", inner_h + 2, inner_w + 2)]
    for quad in _QUADS:
        ph, pw = piece_arrays[quad].shape
        objects.append((quad, ph, pw))
    objects.sort(key=lambda item: item[1] * item[2], reverse=True)

    placements: dict[str, tuple[int, int]] = {}
    for name, oh, ow in objects:
        candidates = [
            (r, c)
            for r in range(1, canvas_h - oh)
            for c in range(1, canvas_w - ow)
            if _can_place_rect(reserved, r, c, oh, ow, pad=1)
        ]
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        placements[name] = (top, left)
        _reserve(reserved, top, left, oh, ow, pad=1)

    frame_top, frame_left = placements["frame"]
    grid[frame_top, frame_left : frame_left + inner_w + 2] = frame_color
    grid[frame_top + inner_h + 1, frame_left : frame_left + inner_w + 2] = frame_color
    grid[frame_top : frame_top + inner_h + 2, frame_left] = frame_color
    grid[frame_top : frame_top + inner_h + 2, frame_left + inner_w + 1] = frame_color

    for quad in _QUADS:
        top, left = placements[quad]
        sprite = piece_arrays[quad]
        h, w = sprite.shape
        patch = grid[top : top + h, left : left + w]
        patch[sprite != 0] = sprite[sprite != 0]

    expected = np.zeros_like(grid)
    expected[frame_top, frame_left : frame_left + inner_w + 2] = frame_color
    expected[frame_top + inner_h + 1, frame_left : frame_left + inner_w + 2] = frame_color
    expected[frame_top : frame_top + inner_h + 2, frame_left] = frame_color
    expected[frame_top : frame_top + inner_h + 2, frame_left + inner_w + 1] = frame_color
    target = expected[frame_top + 1 : frame_top + 1 + inner_h, frame_left + 1 : frame_left + 1 + inner_w]
    target[mask] = piece_color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
