from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SPRITE_COLOR = 2
FRAME_COLOR = 4

GRID_SIDE_RANGE = (20, 29)
FRAME_COUNT_CHOICES = (3, 4)
FRAME_INNER_MAX = 6
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _paste(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = out[top : top + h, left : left + w]
    mask = patch != BG
    target[mask] = patch[mask]


def _transforms(sprite: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "wide": sprite[:, ::-1],
        "tall": sprite[::-1, :],
        "square": np.rot90(sprite, 2),
    }


def _find_sprite(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = _components(grid == SPRITE_COLOR)
    assert len(comps) == 1, "expected exactly one red template component"
    bbox = _bbox_from_coords(comps[0])
    r0, c0, r1, c1 = bbox
    sprite = np.where(grid[r0 : r1 + 1, c0 : c1 + 1] == SPRITE_COLOR, SPRITE_COLOR, BG)
    return sprite, bbox


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        expected_area = 2 * (h + w) - 4
        if len(comp) != expected_area:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == FRAME_COLOR):
            continue
        if np.any(patch[1:-1, 1:-1] == FRAME_COLOR):
            continue
        inner_h = h - 2
        inner_w = w - 2
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )
    assert len(frames) >= 3, "expected at least three yellow hollow rectangles"
    return sorted(frames, key=lambda item: item["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp mirrored copies of the red template into yellow frames according to each frame's inner aspect ratio.

    Input structure:
    - A black background contains one small red connected sprite near the upper-left and several yellow one-pixel-thick hollow rectangular frames elsewhere.
    - The red sprite is asymmetric, uses color 2 only, and its bounding box is at most 3x3.
    - Each yellow frame has an empty interior large enough to hold the red sprite, and the frame's inner width/height relation is the only control signal.

    Rule:
    - Crop the red sprite to its bounding box and treat it as the template.
    - For every yellow frame, inspect the frame interior size excluding the border: if inner width is larger than inner height, horizontally flip the template; if inner height is larger, vertically flip it; if they are equal, rotate it by 180 degrees.
    - Place the transformed template at the top-left corner of that frame interior while leaving the original template and all yellow frame pixels unchanged.

    Key insight:
    - The yellow frame geometry is an implicit instruction channel: the rectangle's inner aspect ratio, not an extra marker, encodes which mirror transform to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprite, _sprite_bbox = _find_sprite(g)
    frames = _find_frames(g)
    transformed = _transforms(sprite)

    out = g.copy()
    for frame in frames:
        r0, c0, _r1, _c1 = frame["inner_bbox"]
        if frame["inner_w"] > frame["inner_h"]:
            patch = transformed["wide"]
        elif frame["inner_h"] > frame["inner_w"]:
            patch = transformed["tall"]
        else:
            patch = transformed["square"]
        _paste(out, patch, r0, c0)
    return out


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _inflate_bbox(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _has_clearance(
    box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], pad: int = 1
) -> bool:
    grown = _inflate_bbox(box, pad)
    return all(not _bbox_intersects(grown, _inflate_bbox(other, pad)) for other in others)


def _build_frame_patch(inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    return patch


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return mask.shape, tuple(int(x) for x in mask.astype(int).ravel())


def _sprite_candidates() -> list[np.ndarray]:
    candidates: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for bits in range(1 << 9):
        area = bits.bit_count()
        if area < 3 or area > 5:
            continue
        raw = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        if len(_components(raw)) != 1:
            continue
        mask = _crop_bool(raw)
        if mask.shape[0] > 3 or mask.shape[1] > 3:
            continue
        lr = mask[:, ::-1]
        ud = mask[::-1, :]
        rot = np.rot90(mask, 2)
        if np.array_equal(mask, lr) or np.array_equal(mask, ud) or np.array_equal(mask, rot):
            continue
        if np.array_equal(lr, ud) or np.array_equal(lr, rot) or np.array_equal(ud, rot):
            continue
        sig = _mask_signature(mask)
        candidates[sig] = mask
    return list(candidates.values())


SPRITE_CANDIDATES = _sprite_candidates()


def _sample_sprite(rng: random.Random) -> np.ndarray:
    mask = SPRITE_CANDIDATES[rng.randrange(len(SPRITE_CANDIDATES))].copy()
    k = rng.randrange(4)
    if k:
        mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        mask = mask[:, ::-1]
    sprite = np.where(mask, SPRITE_COLOR, BG)
    transformed = _transforms(sprite)
    assert not np.array_equal(sprite, transformed["wide"])
    assert not np.array_equal(sprite, transformed["tall"])
    assert not np.array_equal(sprite, transformed["square"])
    assert not np.array_equal(transformed["wide"], transformed["tall"])
    assert not np.array_equal(transformed["wide"], transformed["square"])
    assert not np.array_equal(transformed["tall"], transformed["square"])
    return sprite


def _sample_dims(rng: random.Random, sprite_h: int, sprite_w: int, kind: str) -> tuple[int, int] | None:
    min_h = max(3, sprite_h + 1)
    min_w = max(3, sprite_w + 1)
    if kind == "square":
        start = max(min_h, min_w)
        if start > FRAME_INNER_MAX:
            return None
        side = rng.randint(start, FRAME_INNER_MAX)
        return side, side
    if kind == "wide":
        options = []
        for h in range(min_h, FRAME_INNER_MAX + 1):
            for w in range(max(min_w, h + 1), FRAME_INNER_MAX + 1):
                options.append((h, w))
        return rng.choice(options) if options else None
    if kind == "tall":
        options = []
        for w in range(min_w, FRAME_INNER_MAX + 1):
            for h in range(max(min_h, w + 1), FRAME_INNER_MAX + 1):
                options.append((h, w))
        return rng.choice(options) if options else None
    raise AssertionError("unknown frame kind")


def _place_rectangles(
    rng: random.Random,
    height: int,
    width: int,
    sprite_bbox: tuple[int, int, int, int],
    frame_specs: list[tuple[str, int, int]],
) -> list[tuple[str, int, int, int, int]] | None:
    occupied = [sprite_bbox]
    placed: list[tuple[str, int, int, int, int]] = []
    for kind, outer_h, outer_w in sorted(frame_specs, key=lambda item: item[1] * item[2], reverse=True):
        found = None
        for _ in range(300):
            r0 = rng.randint(0, height - outer_h)
            c0 = rng.randint(0, width - outer_w)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if _has_clearance(bbox, occupied, pad=1):
                found = bbox
                break
        if found is None:
            return None
        occupied.append(found)
        placed.append((kind, found[0], found[1], outer_h, outer_w))
    return placed


def _build_input_from_layout(
    height: int,
    width: int,
    sprite: np.ndarray,
    sprite_top: int,
    sprite_left: int,
    frames: list[tuple[str, int, int, int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    _paste(grid, sprite, sprite_top, sprite_left)
    for kind, top, left, outer_h, outer_w in frames:
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        frame_patch = _build_frame_patch(inner_h, inner_w)
        _paste(grid, frame_patch, top, left)
    return grid


def _validate_input_semantics(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    colors = set(int(v) for v in np.unique(grid))
    assert colors <= {BG, SPRITE_COLOR, FRAME_COLOR}

    sprite, sprite_bbox = _find_sprite(grid)
    sprite_h, sprite_w = sprite.shape
    assert sprite_h <= 3 and sprite_w <= 3
    assert 3 <= int((sprite == SPRITE_COLOR).sum()) <= 5
    assert sprite_bbox[0] <= 3 and sprite_bbox[1] <= 3, "sprite must live near the upper-left"

    transformed = _transforms(sprite)
    assert not np.array_equal(sprite, transformed["wide"])
    assert not np.array_equal(sprite, transformed["tall"])
    assert not np.array_equal(transformed["wide"], transformed["tall"])
    assert not np.array_equal(transformed["wide"], transformed["square"])
    assert not np.array_equal(transformed["tall"], transformed["square"])

    frames = _find_frames(grid)
    assert 3 <= len(frames) <= 4
    kinds = Counter()
    occupied = [sprite_bbox]
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch[1:-1, 1:-1] == BG), "input frame interior must be empty"
        assert frame["inner_h"] >= 3 and frame["inner_w"] >= 3
        assert frame["inner_h"] > sprite_h and frame["inner_w"] > sprite_w
        if frame["inner_w"] > frame["inner_h"]:
            kinds["wide"] += 1
        elif frame["inner_h"] > frame["inner_w"]:
            kinds["tall"] += 1
        else:
            kinds["square"] += 1
        assert _has_clearance(frame["bbox"], occupied, pad=1)
        occupied.append(frame["bbox"])
    assert kinds["wide"] >= 1 and kinds["tall"] >= 1 and kinds["square"] >= 1


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    sprite, sprite_bbox = _find_sprite(inp)
    transformed = _transforms(sprite)
    frames = _find_frames(inp)

    assert np.array_equal(out[sprite_bbox[0] : sprite_bbox[2] + 1, sprite_bbox[1] : sprite_bbox[3] + 1], inp[sprite_bbox[0] : sprite_bbox[2] + 1, sprite_bbox[1] : sprite_bbox[3] + 1])
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1] == FRAME_COLOR, inp[r0 : r1 + 1, c0 : c1 + 1] == FRAME_COLOR)
        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        if frame["inner_w"] > frame["inner_h"]:
            expected = transformed["wide"]
        elif frame["inner_h"] > frame["inner_w"]:
            expected = transformed["tall"]
        else:
            expected = transformed["square"]
        h, w = expected.shape
        patch = out[ir0 : ir0 + h, ic0 : ic0 + w]
        assert np.array_equal(patch, expected)
        interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        mask = np.zeros_like(interior, dtype=bool)
        mask[:h, :w] = expected == SPRITE_COLOR
        assert np.all(interior[~mask] == BG)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_SIDE_RANGE)
    width = rng.randint(*GRID_SIDE_RANGE)

    sprite = _sample_sprite(rng)
    sprite_h, sprite_w = sprite.shape
    sprite_top = rng.randint(0, 2)
    sprite_left = rng.randint(0, 2)
    sprite_bbox = (sprite_top, sprite_left, sprite_top + sprite_h - 1, sprite_left + sprite_w - 1)

    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    frame_kinds = ["wide", "tall", "square"]
    if frame_count == 4:
        frame_kinds.append(rng.choice(frame_kinds))
    rng.shuffle(frame_kinds)

    frame_specs: list[tuple[str, int, int]] = []
    for kind in frame_kinds:
        dims = _sample_dims(rng, sprite_h, sprite_w, kind)
        if dims is None:
            return None
        inner_h, inner_w = dims
        frame_specs.append((kind, inner_h + 2, inner_w + 2))

    layout = _place_rectangles(rng, height, width, sprite_bbox, frame_specs)
    if layout is None:
        return None

    grid = _build_input_from_layout(height, width, sprite, sprite_top, sprite_left, layout)
    try:
        _validate_input_semantics(grid)
        out = main(grid)
        _validate_output_semantics(grid, out)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid implicit-aspect-ratio mirror task")
