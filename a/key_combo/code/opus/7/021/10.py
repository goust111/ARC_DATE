from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
INPUT_COUNT_OPTIONS = (3, 4)
BASE_SCALES = (1, 2, 3)
GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (18, 26)
MIN_GAP = 1
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_nn(proto: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(proto, int(factor), axis=0), int(factor), axis=1)


def _intervals_conflict(a0: int, a1: int, b0: int, b1: int, gap: int) -> bool:
    return not (a1 + gap < b0 or b1 + gap < a0)


def _paste_checked(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise AssertionError("patch placement is out of bounds")
    target = dst[r0 : r0 + h, c0 : c0 + w]
    if np.any(target != bg):
        raise AssertionError("sprites must not overlap")
    target[:] = patch


def _normalize_scaled_sprite(patch: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    h, w = patch.shape
    if h != w or h % 2 != 0 or h == 0:
        raise AssertionError("each sprite must be a non-empty square with even side length")
    scale = h // 2
    if 2 * scale != w:
        raise AssertionError("sprite must be a 2x2 prototype scaled uniformly")

    proto = np.zeros((2, 2), dtype=int)
    for pr in range(2):
        for pc in range(2):
            block = patch[pr * scale : (pr + 1) * scale, pc * scale : (pc + 1) * scale]
            color = int(block[0, 0])
            if color == bg or not np.all(block == color):
                raise AssertionError("every scale block must be a solid non-background color")
            proto[pr, pc] = color
    return proto, int(scale)


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _most_common_color(grid)
    comps = _components(grid != bg)
    if not comps:
        raise AssertionError("expected foreground sprites")

    records = []
    for coords in comps:
        box = _bbox(coords)
        patch = _crop_bbox(grid, box)
        proto, scale = _normalize_scaled_sprite(patch, bg)
        records.append(
            {
                "coords": coords,
                "bbox": box,
                "patch": patch,
                "proto": proto,
                "scale": scale,
            }
        )
    records.sort(key=lambda rec: (rec["bbox"][1], rec["bbox"][0], rec["scale"]))
    return int(bg), records


def _render(
    shape: tuple[int, int],
    bg: int,
    prototype: np.ndarray,
    records: list[dict],
    *,
    mirror_position: bool,
    inverse_scale: bool,
    preserve_center: bool,
) -> np.ndarray | None:
    h, w = shape
    s_max = max(int(rec["scale"]) for rec in records)
    out = np.full((h, w), bg, dtype=int)

    for rec in records:
        r0, c0, _r1, _c1 = rec["bbox"]
        scale = int(rec["scale"])
        new_scale = s_max + 1 - scale if inverse_scale else scale
        new_patch = _scale_nn(prototype, new_scale)

        new_r0 = r0 + scale - new_scale if preserve_center else r0
        old_width = 2 * scale
        new_c0 = w - c0 - old_width if mirror_position else c0

        try:
            _paste_checked(out, new_patch, int(new_r0), int(new_c0), bg)
        except AssertionError:
            return None

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror every scaled sprite to the opposite horizontal position while inverting its scale factor.

    Input structure:
    - The grid has a black background and 3-4 non-overlapping multicolor sprites.
    - Every sprite is a nearest-neighbor scaled copy of the same fully filled 2x2 prototype.
    - Sprite bounding boxes are square, with side lengths 2, 4, or 6, and their vertical centers lie in the same narrow row band.

    Rule:
    - Identify each sprite, recover the shared 2x2 prototype, and read its scale factor S from the bbox size.
    - Let S_max be the largest scale factor present, then replace each sprite by the same prototype at inverse scale S' = S_max + 1 - S.
    - Move each sprite's bbox to the horizontally mirrored column new_c0 = grid_width - old_c0 - old_width, keep the prototype orientation unchanged, and vertically center the resized sprite on the original row-band center.

    Key insight:
    - The task couples two dependencies at once: bbox positions are mirrored using the original sprite widths, but the drawn sprite sizes are inverted, so correct placement requires recovering both the shared prototype and each instance's scale.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    bg, records = _extract_sprites(g)
    if not 3 <= len(records) <= 4:
        raise AssertionError("expected 3 or 4 sprites")

    prototype = records[0]["proto"]
    if any(not np.array_equal(rec["proto"], prototype) for rec in records[1:]):
        raise AssertionError("all sprites must share the same 2x2 prototype")

    out = _render(
        g.shape,
        bg,
        prototype,
        records,
        mirror_position=True,
        inverse_scale=True,
        preserve_center=True,
    )
    if out is None:
        raise AssertionError("mirrored inverse-scale layout must fit without overlap")
    return out


def _random_prototype() -> np.ndarray:
    while True:
        n_colors = random.choice((2, 3))
        colors = random.sample(PALETTE, n_colors)
        proto = np.array(random.choices(colors, k=4), dtype=int).reshape(2, 2)
        if len(set(map(int, proto.ravel()))) != n_colors:
            continue
        if np.array_equal(proto, np.fliplr(proto)):
            continue
        if np.array_equal(proto, np.flipud(proto)):
            continue
        if np.array_equal(proto, proto.T):
            continue
        return proto


def _sample_scales() -> list[int]:
    count = random.choice(INPUT_COUNT_OPTIONS)
    scales = list(BASE_SCALES)
    if count == 4:
        scales.append(random.choices([1, 2, 3], weights=[2, 3, 2], k=1)[0])
    random.shuffle(scales)
    return scales


def _find_columns(width: int, scales: list[int], s_max: int, gap: int) -> list[int] | None:
    order = list(range(len(scales)))
    order.sort(key=lambda i: (-max(scales[i], s_max + 1 - scales[i]), -scales[i], random.random()))
    chosen: dict[int, int] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True

        idx = order[pos]
        scale = int(scales[idx])
        old_w = 2 * scale
        new_scale = s_max + 1 - scale
        new_w = 2 * new_scale
        candidates = list(range(0, width - old_w + 1))
        random.shuffle(candidates)

        for c0 in candidates:
            out_c0 = width - c0 - old_w
            if out_c0 < 0 or out_c0 + new_w > width:
                continue

            ok = True
            for other_idx, other_c0 in chosen.items():
                other_scale = int(scales[other_idx])
                other_old_w = 2 * other_scale
                other_new_w = 2 * (s_max + 1 - other_scale)
                other_out_c0 = width - other_c0 - other_old_w

                if _intervals_conflict(c0, c0 + old_w - 1, other_c0, other_c0 + other_old_w - 1, gap):
                    ok = False
                    break
                if _intervals_conflict(out_c0, out_c0 + new_w - 1, other_out_c0, other_out_c0 + other_new_w - 1, gap):
                    ok = False
                    break

            if ok:
                chosen[idx] = c0
                if backtrack(pos + 1):
                    return True
                del chosen[idx]
        return False

    if not backtrack(0):
        return None
    return [int(chosen[i]) for i in range(len(scales))]


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        bg, records = _extract_sprites(grid)
        if bg != BG or len(records) not in INPUT_COUNT_OPTIONS:
            return False

        prototype = records[0]["proto"]
        if any(not np.array_equal(rec["proto"], prototype) for rec in records[1:]):
            return False

        scales = [int(rec["scale"]) for rec in records]
        if not {1, 2, 3}.issubset(set(scales)):
            return False
        if len(records) == 3 and sorted(scales) != [1, 2, 3]:
            return False

        centers = {2 * rec["bbox"][0] + 2 * rec["scale"] - 1 for rec in records}
        if len(centers) != 1:
            return False

        transformed = main(grid)
        expected = _render(
            grid.shape,
            bg,
            prototype,
            records,
            mirror_position=True,
            inverse_scale=True,
            preserve_center=True,
        )
        if expected is None or not np.array_equal(transformed, expected):
            return False

        mirror_only = _render(
            grid.shape,
            bg,
            prototype,
            records,
            mirror_position=True,
            inverse_scale=False,
            preserve_center=True,
        )
        if mirror_only is not None and np.array_equal(transformed, mirror_only):
            return False

        scale_only = _render(
            grid.shape,
            bg,
            prototype,
            records,
            mirror_position=False,
            inverse_scale=True,
            preserve_center=True,
        )
        if scale_only is not None and np.array_equal(transformed, scale_only):
            return False

        top_preserving = _render(
            grid.shape,
            bg,
            prototype,
            records,
            mirror_position=True,
            inverse_scale=True,
            preserve_center=False,
        )
        if top_preserving is not None and np.array_equal(transformed, top_preserving):
            return False

        out_bg, out_records = _extract_sprites(transformed)
        if out_bg != BG or len(out_records) != len(records):
            return False
        if any(not np.array_equal(rec["proto"], prototype) for rec in out_records):
            return False

        out_scales = sorted(int(rec["scale"]) for rec in out_records)
        expected_scales = sorted(max(scales) + 1 - s for s in scales)
        if out_scales != expected_scales:
            return False

        if np.array_equal(grid, transformed):
            return False

    except Exception:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(*GRID_HEIGHT_RANGE)
        w = random.randint(*GRID_WIDTH_RANGE)
        proto = _random_prototype()
        scales = _sample_scales()
        s_max = max(scales)

        if w < sum(2 * s for s in scales) + MIN_GAP * (len(scales) - 1):
            continue

        band_row = random.randint(s_max + 1, h - s_max - 1)
        cols = _find_columns(w, scales, s_max, MIN_GAP)
        if cols is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        for scale, c0 in zip(scales, cols, strict=True):
            r0 = band_row - scale
            patch = _scale_nn(proto, scale)
            grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

        if _validate_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid mirrored inverse-scale ARC input")
