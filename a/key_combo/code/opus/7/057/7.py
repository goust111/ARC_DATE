from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PINK = 6

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def compress_mask(mask: np.ndarray) -> np.ndarray:
    mask = crop_mask(np.asarray(mask, dtype=bool))
    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    mask = mask[row_keep, :]

    col_keep = [0]
    for c in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, c], mask[:, c - 1]):
            col_keep.append(c)
    return mask[:, col_keep]


def mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    compressed = compress_mask(mask)
    return tuple(tuple(int(v) for v in row) for row in compressed)


def dihedral_signatures(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    sigs: set[tuple[tuple[int, ...], ...]] = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        sigs.add(mask_signature(rotated))
        sigs.add(mask_signature(np.fliplr(rotated)))
    return sigs


def is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = component_mask(coords)
    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def frame_components(grid: np.ndarray, color: int) -> list[dict]:
    frames: list[dict] = []
    for coords in components_mask(grid == color):
        if not is_rectangular_frame(coords):
            continue
        b = bbox_coords(coords)
        area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        frames.append({"coords": coords, "bbox": b, "area": int(area)})
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Use the red shape inside the largest grey frame as a prototype, then keep only blue outside objects whose run-compressed topology matches it up to rotation or reflection.

    Input structure:
    - A black background contains 2-3 complete one-pixel-thick grey rectangular frames.
    - The unique largest grey frame encloses a red 4-connected prototype shape; smaller frames may contain only black or a few pink noise pixels.
    - Several blue 4-connected candidate objects lie outside all frames.

    Rule:
    - Identify all grey rectangular frames and choose the unique largest by bounding-box area.
    - Crop the interior of that frame, extract the red prototype, crop it to its own bbox, and run-compress adjacent identical rows and columns.
    - For each blue candidate object, crop to bbox, run-compress the binary mask, and compare it against the prototype under all 8 rotation/reflection variants.
    - Recolor matching candidates to green and erase non-matching candidates to black, leaving the frames and their contents unchanged.

    Key insight:
    - The prototype is hidden inside the largest frame, and stretched candidate variants only become comparable after compressing repeated row and column runs before testing dihedral shape equivalence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = sorted(frame_components(g, GREY), key=lambda item: item["area"], reverse=True)
    assert len(frames) >= 2, "expected at least two grey frames"
    if len(frames) > 1:
        assert frames[0]["area"] > frames[1]["area"], "largest frame must be unique"

    r0, c0, r1, c1 = frames[0]["bbox"]
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    proto_coords = np.argwhere(interior == RED)
    assert proto_coords.size > 0, "largest frame must contain a red prototype"
    prototype = component_mask(proto_coords)
    prototype_sigs = dihedral_signatures(prototype)

    out = g.copy()
    for coords in components_mask(g == BLUE):
        mask = component_mask(coords)
        if mask_signature(mask) in prototype_sigs:
            out[coords[:, 0], coords[:, 1]] = GREEN
        else:
            out[coords[:, 0], coords[:, 1]] = BLACK
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = min(rs)
    c0 = min(cs)
    shifted = sorted((r - r0, c - c0) for r, c in cells)
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _random_connected_mask(area_lo: int, area_hi: int, max_side: int) -> np.ndarray:
    for _ in range(200):
        target = random.randint(area_lo, area_hi)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(random.choice(frontier))
        mask = _normalize_cells(cells)
        if mask.shape[0] > max_side or mask.shape[1] > max_side:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == mask.size:
            continue
        return mask
    raise RuntimeError("failed to build a connected mask")


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = (variant.tobytes(), tuple(map(int, variant.shape)))
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _stretch_mask(mask: np.ndarray, max_area: int) -> np.ndarray:
    for _ in range(200):
        variant = random.choice(_orientations(mask))
        row_factors = [random.choice((1, 1, 2)) for _ in range(variant.shape[0])]
        col_factors = [random.choice((1, 1, 2)) for _ in range(variant.shape[1])]
        stretched = np.repeat(np.repeat(variant, row_factors, axis=0), col_factors, axis=1)
        area = int(stretched.sum())
        if not (3 <= area <= max_area):
            continue
        if stretched.shape[0] > 6 or stretched.shape[1] > 6:
            continue
        return stretched
    raise RuntimeError("failed to stretch a matching candidate")


def _random_distractor(proto_sigs: set[tuple[tuple[int, ...], ...]]) -> np.ndarray:
    for _ in range(300):
        mask = _random_connected_mask(3, 8, 6)
        if mask_signature(mask) not in proto_sigs:
            return mask
    raise RuntimeError("failed to build a distractor")


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_rect(
    h: int,
    w: int,
    height: int,
    width: int,
    existing: list[tuple[int, int, int, int]],
    pad: int,
) -> tuple[int, int, int, int] | None:
    candidates = [
        (r0, c0, r0 + height - 1, c0 + width - 1)
        for r0 in range(0, h - height + 1)
        for c0 in range(0, w - width + 1)
    ]
    random.shuffle(candidates)
    for rect in candidates:
        if all(not _rect_intersects(rect, other, pad) for other in existing):
            return rect
    return None


def _paint_frame(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = color
        cells.append((rr, cc))
    return cells


def _place_prototype(grid: np.ndarray, frame: tuple[int, int, int, int], proto: np.ndarray) -> None:
    r0, c0, r1, c1 = frame
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    proto_h, proto_w = proto.shape
    assert proto_h <= inner_h and proto_w <= inner_w

    margin_r = 1 if inner_h - proto_h >= 2 else 0
    margin_c = 1 if inner_w - proto_w >= 2 else 0
    rel_r = random.randint(margin_r, inner_h - proto_h - margin_r)
    rel_c = random.randint(margin_c, inner_w - proto_w - margin_c)
    _paint_mask(grid, proto, r0 + 1 + rel_r, c0 + 1 + rel_c, RED)


def _add_small_frame_noise(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if not cells:
        return
    noise_count = random.choice((0, 0, 1, 1, 2))
    if noise_count == 0:
        return
    for r, c in random.sample(cells, k=min(noise_count, len(cells))):
        grid[r, c] = PINK


def _candidate_specs(proto: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    total = random.randint(4, 6)
    match_count = random.randint(2, total - 1)
    proto_sigs = dihedral_signatures(proto)
    specs: list[tuple[np.ndarray, bool]] = []
    for _ in range(match_count):
        specs.append((_stretch_mask(proto, max_area=8), True))
    for _ in range(total - match_count):
        specs.append((_random_distractor(proto_sigs), False))
    random.shuffle(specs)
    return specs


def _validate_generated(grid: np.ndarray, candidate_cells: list[tuple[list[tuple[int, int]], bool]]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    matched = 0
    unmatched = 0
    for cells, should_match in candidate_cells:
        values = {int(out[r, c]) for r, c in cells}
        if should_match:
            if values != {GREEN}:
                return False
            matched += 1
        else:
            if values != {BLACK}:
                return False
            unmatched += 1
    return matched >= 2 and unmatched >= 1


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(19, 24)
        w = random.randint(19, 24)
        grid = np.full((h, w), BLACK, dtype=int)

        proto = _random_connected_mask(3, 6, 4)
        large_h = random.randint(max(7, proto.shape[0] + 4), 10)
        large_w = random.randint(max(7, proto.shape[1] + 4), 10)

        frame_count = random.randint(2, 3)
        small_dims: list[tuple[int, int]] = []
        ok = True
        for _ in range(frame_count - 1):
            choices = [
                (fh, fw)
                for fh in range(4, 8)
                for fw in range(4, 8)
                if fh * fw < large_h * large_w
            ]
            if not choices:
                ok = False
                break
            small_dims.append(random.choice(choices))
        if not ok:
            continue

        rects: list[tuple[int, int, int, int]] = []
        large_rect = _place_rect(h, w, large_h, large_w, rects, pad=1)
        if large_rect is None:
            continue
        rects.append(large_rect)
        small_rects: list[tuple[int, int, int, int]] = []
        failed = False
        for fh, fw in small_dims:
            rect = _place_rect(h, w, fh, fw, rects, pad=1)
            if rect is None:
                failed = True
                break
            rects.append(rect)
            small_rects.append(rect)
        if failed:
            continue

        _paint_frame(grid, large_rect)
        for rect in small_rects:
            _paint_frame(grid, rect)

        _place_prototype(grid, large_rect, proto)
        for rect in small_rects:
            _add_small_frame_noise(grid, rect)

        reserved = rects[:]
        candidate_cells: list[tuple[list[tuple[int, int]], bool]] = []
        for mask, should_match in _candidate_specs(proto):
            ch, cw = mask.shape
            rect = _place_rect(h, w, ch, cw, reserved, pad=1)
            if rect is None:
                failed = True
                break
            reserved.append(rect)
            cells = _paint_mask(grid, mask, rect[0], rect[1], BLUE)
            candidate_cells.append((cells, should_match))
        if failed:
            continue

        if _validate_generated(grid, candidate_cells):
            return grid

    raise RuntimeError("failed to generate a valid ARC sample")
