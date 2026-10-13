from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GRAY = 5
DIR_ORDER = ("top", "right", "bottom", "left")
DIR_STEPS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR_STEPS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_mask(h: int, w: int, t: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:t, :] = True
    mask[-t:, :] = True
    mask[:, :t] = True
    mask[:, -t:] = True
    return mask


def _rectangular(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _marker_position(shape: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = map(int, shape)
    if side == "top":
        assert w % 2 == 1
        return 0, w // 2
    if side == "bottom":
        assert w % 2 == 1
        return h - 1, w // 2
    if side == "left":
        assert h % 2 == 1
        return h // 2, 0
    if side == "right":
        assert h % 2 == 1
        return h // 2, w - 1
    raise AssertionError(f"bad side {side}")


def _marker_side(patch: np.ndarray) -> str | None:
    coords = np.argwhere(patch == GRAY)
    if coords.shape[0] != 1:
        return None
    r, c = map(int, coords[0])
    h, w = patch.shape
    hits = []
    if w % 2 == 1 and r == 0 and c == w // 2:
        hits.append("top")
    if w % 2 == 1 and r == h - 1 and c == w // 2:
        hits.append("bottom")
    if h % 2 == 1 and c == 0 and r == h // 2:
        hits.append("left")
    if h % 2 == 1 and c == w - 1 and r == h // 2:
        hits.append("right")
    return hits[0] if len(hits) == 1 else None


def _extract_stamp(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = set(map(int, np.unique(patch))) - {BG}
    if GRAY not in colors or len(colors) != 2:
        return None
    if np.any(patch == BG):
        return None
    marker_side = _marker_side(patch)
    if marker_side is None:
        return None
    stamp_color = next(color for color in colors if color != GRAY)
    return {
        "bbox": (r0, c0, r1, c1),
        "shape": tuple(map(int, patch.shape)),
        "patch": patch.copy(),
        "marker_side": marker_side,
        "color": int(stamp_color),
    }


def _extract_mold(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = set(map(int, np.unique(patch))) - {BG}
    if len(colors) != 1 or GRAY in colors:
        return None
    color = next(iter(colors))
    mask = patch == color
    h, w = map(int, patch.shape)
    matches = []
    for t in range(1, min(h, w) // 2 + 1):
        full = _ring_mask(h, w, t)
        if np.any(mask & ~full):
            continue
        missing = full & ~mask
        if not bool(missing.any()) or not _rectangular(missing):
            continue
        mr0, mc0, mr1, mc1 = _bbox(np.argwhere(missing))
        side = None
        if mr0 == 0 and mr1 == t - 1 and mc0 >= t and mc1 <= w - t - 1:
            side = "top"
        elif mr0 == h - t and mr1 == h - 1 and mc0 >= t and mc1 <= w - t - 1:
            side = "bottom"
        elif mc0 == 0 and mc1 == t - 1 and mr0 >= t and mr1 <= h - t - 1:
            side = "left"
        elif mc0 == w - t and mc1 == w - 1 and mr0 >= t and mr1 <= h - t - 1:
            side = "right"
        if side is None:
            continue
        matches.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": int(color),
                "thickness": int(t),
                "gap_side": side,
                "gap_bbox": (r0 + mr0, c0 + mc0, r0 + mr1, c0 + mc1),
                "gap_shape": (mr1 - mr0 + 1, mc1 - mc0 + 1),
            }
        )
    if len(matches) != 1:
        return None
    return matches[0]


def _size_key(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    return (min(h, w), max(h, w))


def _oriented_stamp(stamp_patch: np.ndarray, target_shape: tuple[int, int], target_side: str) -> np.ndarray:
    candidates = []
    for k in range(4):
        rotated = np.rot90(stamp_patch, k)
        if rotated.shape != tuple(target_shape):
            continue
        if _marker_side(rotated) == target_side:
            candidates.append(rotated.copy())
    assert len(candidates) == 1, "stamp orientation must be unique"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each stamp to the mold notch with the same size, rotate the stamp so
    its gray marker points outward, then recolor the stamp to the mold color.
    """
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)

    molds: list[dict] = []
    stamps: list[dict] = []
    for coords in comps:
        stamp = _extract_stamp(g, coords)
        if stamp is not None:
            stamps.append(stamp)
            continue
        mold = _extract_mold(g, coords)
        if mold is not None:
            molds.append(mold)
            continue
        raise AssertionError("unrecognized component")

    assert molds and stamps and len(molds) == len(stamps)

    mold_by_key: dict[tuple[int, int], dict] = {}
    stamp_by_key: dict[tuple[int, int], dict] = {}
    for mold in molds:
        key = _size_key(mold["gap_shape"])
        assert key not in mold_by_key, "mold gap sizes must be unique"
        mold_by_key[key] = mold
    for stamp in stamps:
        key = _size_key(stamp["shape"])
        assert key not in stamp_by_key, "stamp sizes must be unique"
        stamp_by_key[key] = stamp
    assert set(mold_by_key) == set(stamp_by_key), "sizes must pair one-to-one"

    out = g.copy()
    for stamp in stamps:
        r0, c0, r1, c1 = stamp["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for key, mold in mold_by_key.items():
        stamp = stamp_by_key[key]
        rotated = _oriented_stamp(stamp["patch"], mold["gap_shape"], mold["gap_side"])
        recolored = rotated.copy()
        recolored[(recolored != BG) & (recolored != GRAY)] = mold["color"]
        gr0, gc0, gr1, gc1 = mold["gap_bbox"]
        out[gr0 : gr1 + 1, gc0 : gc1 + 1] = recolored

    return out


SIZE_CHOICES = [(3, 3), (3, 5), (3, 7), (3, 9), (5, 5), (5, 7)]
PAIR_COUNT_RANGE = (2, 4)
INNER_SHORT_RANGE = (3, 7)
INNER_SLACK_RANGE = (0, 4)
PLACEMENT_PADDING = 2
PLACEMENT_TRIES = 300
GENERATION_TRIES = 500
CANVAS_MARGIN = 1
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _build_mold(size_key: tuple[int, int], side: str, color: int) -> dict:
    thickness, span = map(int, size_key)
    if side in {"top", "bottom"}:
        inner_h = random.randint(*INNER_SHORT_RANGE)
        inner_w = span + random.randint(*INNER_SLACK_RANGE)
        outer_h = inner_h + 2 * thickness
        outer_w = inner_w + 2 * thickness
        offset = random.randint(0, inner_w - span)
        gap_r0 = 0 if side == "top" else outer_h - thickness
        gap_c0 = thickness + offset
        gap_h = thickness
        gap_w = span
    else:
        inner_h = span + random.randint(*INNER_SLACK_RANGE)
        inner_w = random.randint(*INNER_SHORT_RANGE)
        outer_h = inner_h + 2 * thickness
        outer_w = inner_w + 2 * thickness
        offset = random.randint(0, inner_h - span)
        gap_r0 = thickness + offset
        gap_c0 = 0 if side == "left" else outer_w - thickness
        gap_h = span
        gap_w = thickness

    patch = np.full((outer_h, outer_w), BG, dtype=int)
    patch[_ring_mask(outer_h, outer_w, thickness)] = color
    patch[gap_r0 : gap_r0 + gap_h, gap_c0 : gap_c0 + gap_w] = BG
    return {
        "patch": patch,
        "gap_bbox_local": (gap_r0, gap_c0, gap_r0 + gap_h - 1, gap_c0 + gap_w - 1),
        "gap_shape": (gap_h, gap_w),
        "gap_side": side,
        "color": int(color),
    }


def _build_stamp(shape: tuple[int, int], side: str, color: int) -> np.ndarray:
    patch = np.full(shape, int(color), dtype=int)
    rr, cc = _marker_position(shape, side)
    patch[rr, cc] = GRAY
    return patch


def _random_canvas(objects: list[np.ndarray]) -> tuple[int, int]:
    padded_area = sum((patch.shape[0] + 2 * PLACEMENT_PADDING) * (patch.shape[1] + 2 * PLACEMENT_PADDING) for patch in objects)
    max_h = max(int(patch.shape[0]) for patch in objects)
    max_w = max(int(patch.shape[1]) for patch in objects)
    base = int(np.ceil(np.sqrt(padded_area)))
    min_h = max(max_h + 2 * CANVAS_MARGIN, base)
    min_w = max(max_w + 2 * CANVAS_MARGIN, base)
    for _ in range(80):
        h = random.randint(min_h, min_h + 10)
        w = random.randint(min_w, min_w + 10)
        if h * w >= int(1.2 * padded_area):
            return h, w
    return min_h + 6, min_w + 6


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_objects(patches: list[np.ndarray], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    boxes: list[tuple[int, int, int, int]] = []
    for idx in order:
        ph, pw = map(int, patches[idx].shape)
        if ph + 2 * CANVAS_MARGIN > h or pw + 2 * CANVAS_MARGIN > w:
            return None
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(CANVAS_MARGIN, h - ph - CANVAS_MARGIN)
            c0 = random.randint(CANVAS_MARGIN, w - pw - CANVAS_MARGIN)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_too_close(box, other, PLACEMENT_PADDING) for other in boxes):
                continue
            boxes.append(box)
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    assert all(pos is not None for pos in positions)
    return [(int(r), int(c)) for r, c in positions]  # type: ignore[misc]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    assert not bool(np.any(region[mask] != BG))
    region[mask] = patch[mask]


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    size_keys = random.sample(SIZE_CHOICES, n_pairs)
    colors = random.sample(COLOR_POOL, 2 * n_pairs)

    pairs = []
    object_patches: list[np.ndarray] = []
    object_meta: list[tuple[int, str]] = []

    for idx, size_key in enumerate(size_keys):
        mold_color = colors[idx]
        stamp_color = colors[n_pairs + idx]
        side = random.choice(DIR_ORDER)
        mold = _build_mold(size_key, side, mold_color)

        final_stamp = _build_stamp(mold["gap_shape"], side, stamp_color)
        initial_stamp = np.rot90(final_stamp, -random.randint(0, 3))

        pairs.append(
            {
                "mold": mold,
                "final_stamp": final_stamp,
                "initial_stamp": initial_stamp,
            }
        )
        object_patches.append(mold["patch"])
        object_meta.append((idx, "mold"))
        object_patches.append(initial_stamp)
        object_meta.append((idx, "stamp"))

    canvas_h, canvas_w = _random_canvas(object_patches)
    positions = _place_objects(object_patches, canvas_h, canvas_w)
    if positions is None:
        raise ValueError("placement failed")

    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    expected = np.full((canvas_h, canvas_w), BG, dtype=int)
    for patch, (top, left) in zip(object_patches, positions):
        _paste_patch(inp, patch, top, left)
        _paste_patch(expected, patch, top, left)

    for (pair_id, kind), (top, left) in zip(object_meta, positions):
        pair = pairs[pair_id]
        if kind == "stamp":
            patch = pair["initial_stamp"]
            h, w = patch.shape
            expected[top : top + h, left : left + w] = BG
        else:
            gap_r0, gap_c0, gap_r1, gap_c1 = pair["mold"]["gap_bbox_local"]
            stamp = pair["final_stamp"].copy()
            stamp[(stamp != BG) & (stamp != GRAY)] = pair["mold"]["color"]
            expected[top + gap_r0 : top + gap_r1 + 1, left + gap_c0 : left + gap_c1 + 1] = stamp

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        try:
            inp, expected = _generate_case()
        except ValueError:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
