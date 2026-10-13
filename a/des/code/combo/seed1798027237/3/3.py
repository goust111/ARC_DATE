from __future__ import annotations

import random

import numpy as np


BG = 0
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _symmetry_kind(mask: np.ndarray) -> str:
    m = np.asarray(mask, dtype=bool)
    vertical = bool(np.array_equal(m, np.fliplr(m)))
    horizontal = bool(np.array_equal(m, np.flipud(m)))
    assert vertical ^ horizontal, "expected exactly one horizontal/vertical symmetry axis"
    return "V" if vertical else "H"


def _side_colors(patch: np.ndarray, kind: str) -> tuple[int, int]:
    mask = patch != BG
    colors = {int(v) for v in np.unique(patch) if int(v) != BG}
    assert len(colors) == 2, "object must use exactly two non-background colors"

    h, w = patch.shape
    if kind == "V":
        mid = w // 2
        left = patch[:, :mid][mask[:, :mid]]
        right_start = mid + (w % 2)
        right = patch[:, right_start:][mask[:, right_start:]]
    else:
        mid = h // 2
        left = patch[:mid, :][mask[:mid, :]]
        right_start = mid + (h % 2)
        right = patch[right_start:, :][mask[right_start:, :]]

    first_colors = {int(v) for v in left}
    second_colors = {int(v) for v in right}
    assert len(first_colors) == 1 and len(second_colors) == 1

    first = next(iter(first_colors))
    second = next(iter(second_colors))
    assert first != second
    return first, second


def _retained_mask(patch: np.ndarray, kind: str, anchor_color: int) -> np.ndarray:
    mask = patch != BG
    keep = np.zeros_like(mask, dtype=bool)
    first_color, second_color = _side_colors(patch, kind)

    if kind == "V":
        h, w = patch.shape
        mid = w // 2
        if anchor_color == first_color:
            keep[:, :mid] = mask[:, :mid]
        else:
            assert anchor_color == second_color
            keep[:, mid + (w % 2) :] = mask[:, mid + (w % 2) :]
        if w % 2:
            keep[:, mid] = patch[:, mid] == anchor_color
    else:
        h, w = patch.shape
        mid = h // 2
        if anchor_color == first_color:
            keep[:mid, :] = mask[:mid, :]
        else:
            assert anchor_color == second_color
            keep[mid + (h % 2) :, :] = mask[mid + (h % 2) :, :]
        if h % 2:
            keep[mid, :] = patch[mid, :] == anchor_color

    return keep


def _find_anchor(
    grid: np.ndarray,
    patch: np.ndarray,
    bbox: tuple[int, int, int, int],
    anchors: list[tuple[int, int]],
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h, w = patch.shape
    assert h == r1 - r0 + 1 and w == c1 - c0 + 1

    candidates: list[tuple[int, int]] = []
    for ar, ac in anchors:
        if ar == r0 - 2 and c0 <= ac <= c1 and r0 - 1 >= 0:
            rel_c = ac - c0
            if grid[r0 - 1, ac] == BG and patch[0, rel_c] != BG:
                candidates.append((ar, ac))
        if ar == r1 + 2 and c0 <= ac <= c1 and r1 + 1 < grid.shape[0]:
            rel_c = ac - c0
            if grid[r1 + 1, ac] == BG and patch[h - 1, rel_c] != BG:
                candidates.append((ar, ac))
        if ac == c0 - 2 and r0 <= ar <= r1 and c0 - 1 >= 0:
            rel_r = ar - r0
            if grid[ar, c0 - 1] == BG and patch[rel_r, 0] != BG:
                candidates.append((ar, ac))
        if ac == c1 + 2 and r0 <= ar <= r1 and c1 + 1 < grid.shape[1]:
            rel_r = ar - r0
            if grid[ar, c1 + 1] == BG and patch[rel_r, w - 1] != BG:
                candidates.append((ar, ac))

    assert len(candidates) == 1, "each object must have exactly one nearby singleton anchor"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each symmetric two-color object along its unique reflection axis and
    keep only the half whose color matches the nearby singleton anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BG)
    objects: list[dict[str, object]] = []
    anchors: list[tuple[int, int]] = []

    for coords in comps:
        if int(coords.shape[0]) == 1:
            anchors.append((int(coords[0, 0]), int(coords[0, 1])))
            continue

        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        mask = patch != BG
        assert int(mask.sum()) == int(coords.shape[0])
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "mask": mask,
            }
        )

    assert objects, "expected at least one symmetric object"
    assert len(objects) == len(anchors), "expected one singleton anchor per object"

    out = g.copy()
    used_anchors: set[tuple[int, int]] = set()

    for obj in objects:
        bbox = obj["bbox"]
        patch = obj["patch"]
        mask = obj["mask"]
        assert isinstance(bbox, tuple)
        assert isinstance(patch, np.ndarray)
        assert isinstance(mask, np.ndarray)

        kind = _symmetry_kind(mask)
        anchor = _find_anchor(g, patch, bbox, anchors)
        assert anchor not in used_anchors
        used_anchors.add(anchor)

        ar, ac = anchor
        anchor_color = int(g[ar, ac])
        object_colors = {int(v) for v in patch[mask]}
        assert anchor_color in object_colors

        keep = _retained_mask(patch, kind, anchor_color)
        r0, c0, r1, c1 = bbox
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[:] = BG
        target[keep] = patch[keep]
        out[ar, ac] = BG

    assert len(used_anchors) == len(anchors)
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (21, 30)
GRID_WIDTH_RANGE = (21, 30)
OBJECT_COUNT_OPTIONS = (2, 3)
PROFILE_LENGTH_RANGE = (4, 8)
HALF_SPAN_RANGE = (2, 5)
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _random_profile(length: int, half_span: int) -> list[int]:
    for _ in range(100):
        profile = [random.randint(1, half_span) for _ in range(length)]
        peak_idx = random.randrange(length)
        profile[peak_idx] = half_span
        if half_span > 1:
            low_idx = random.randrange(length)
            if low_idx == peak_idx and length > 1:
                low_idx = (low_idx + 1) % length
            profile[low_idx] = 1
        if profile != profile[::-1]:
            return profile
    raise RuntimeError("failed to sample a non-palindromic profile")


def _mask_from_profile(profile: list[int], half_span: int, odd_axis: bool) -> np.ndarray:
    length = len(profile)
    width = 2 * half_span + int(odd_axis)
    mask = np.zeros((length, width), dtype=bool)

    for r, extent in enumerate(profile):
        mask[r, half_span - extent : half_span] = True
        if odd_axis:
            mask[r, half_span] = True
        right_start = half_span + int(odd_axis)
        mask[r, right_start : right_start + extent] = True

    return mask


def _random_mask(kind: str) -> np.ndarray:
    for _ in range(100):
        odd_axis = bool(random.getrandbits(1))
        length = random.randint(*PROFILE_LENGTH_RANGE)
        half_span = random.randint(*HALF_SPAN_RANGE)
        profile = _random_profile(length, half_span)

        if kind == "V":
            mask = _mask_from_profile(profile, half_span, odd_axis)
        else:
            mask = _mask_from_profile(profile, half_span, odd_axis).T

        if _symmetry_kind(mask) == kind:
            return mask

    raise RuntimeError("failed to sample a valid symmetric mask")


def _paint_object(mask: np.ndarray, kind: str, first_color: int, second_color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    h, w = mask.shape

    if kind == "V":
        mid = w // 2
        left_mask = np.zeros_like(mask, dtype=bool)
        left_mask[:, :mid] = mask[:, :mid]
        right_mask = np.zeros_like(mask, dtype=bool)
        right_mask[:, mid + (w % 2) :] = mask[:, mid + (w % 2) :]
        patch[left_mask] = first_color
        patch[right_mask] = second_color
        if w % 2:
            for r in range(h):
                if mask[r, mid]:
                    patch[r, mid] = random.choice((first_color, second_color))
    else:
        mid = h // 2
        top_mask = np.zeros_like(mask, dtype=bool)
        top_mask[:mid, :] = mask[:mid, :]
        bottom_mask = np.zeros_like(mask, dtype=bool)
        bottom_mask[mid + (h % 2) :, :] = mask[mid + (h % 2) :, :]
        patch[top_mask] = first_color
        patch[bottom_mask] = second_color
        if h % 2:
            for c in range(w):
                if mask[mid, c]:
                    patch[mid, c] = random.choice((first_color, second_color))

    return patch


def _anchor_offset(mask: np.ndarray, side: str) -> tuple[int, int]:
    h, w = mask.shape
    if side == "up":
        cols = [c for c in range(w) if mask[0, c]]
        return -2, random.choice(cols)
    if side == "down":
        cols = [c for c in range(w) if mask[h - 1, c]]
        return h + 1, random.choice(cols)
    if side == "left":
        rows = [r for r in range(h) if mask[r, 0]]
        return random.choice(rows), -2

    rows = [r for r in range(h) if mask[r, w - 1]]
    return random.choice(rows), w + 1


def _build_bundle() -> tuple[np.ndarray, np.ndarray]:
    kind = random.choice(("V", "H"))
    mask = _random_mask(kind)
    first_color, second_color = random.sample(PALETTE, 2)
    patch = _paint_object(mask, kind, first_color, second_color)
    anchor_color = random.choice((first_color, second_color))
    side = random.choice(("up", "down", "left", "right"))

    h, w = patch.shape
    inp = np.zeros((h + 4, w + 4), dtype=int)
    out = np.zeros_like(inp)
    inp[2 : 2 + h, 2 : 2 + w] = patch

    keep = _retained_mask(patch, kind, anchor_color)
    out_patch = np.zeros_like(patch)
    out_patch[keep] = patch[keep]
    out[2 : 2 + h, 2 : 2 + w] = out_patch

    dr, dc = _anchor_offset(mask, side)
    ar = 2 + dr
    ac = 2 + dc
    assert 0 <= ar < inp.shape[0] and 0 <= ac < inp.shape[1]
    inp[ar, ac] = anchor_color
    return inp, out


def _placeable_positions(occupied: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = occupied.shape
    bh, bw = shape
    positions: list[tuple[int, int]] = []

    for r in range(h - bh + 1):
        for c in range(w - bw + 1):
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(h, r + bh + 1)
            c1 = min(w, c + bw + 1)
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))

    return positions


def _stamp(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    region = dst[top : top + patch.shape[0], left : left + patch.shape[1]]
    region[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    count = random.choice(OBJECT_COUNT_OPTIONS)

    bundles = [_build_bundle() for _ in range(count)]
    bundles.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)

    if sum(inp.size for inp, _ in bundles) > (h * w * 3) // 5:
        return None

    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for bundle_inp, bundle_out in bundles:
        positions = _placeable_positions(occupied, bundle_inp.shape)
        if not positions:
            return None

        top, left = random.choice(positions)
        _stamp(inp, bundle_inp, top, left)
        _stamp(expected, bundle_out, top, left)

        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(h, top + bundle_inp.shape[0] + 1)
        c1 = min(w, left + bundle_inp.shape[1] + 1)
        occupied[r0:r1, c0:c1] = True

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
