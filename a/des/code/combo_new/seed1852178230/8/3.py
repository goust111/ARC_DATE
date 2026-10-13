from __future__ import annotations

from collections import deque
import random
import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = NEIGHBORS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

BG = 0
PALETTE = tuple(range(1, 10))
BRANCHES = ("identity", "hflip", "vflip", "hvflip")
BRANCH_OFFSETS = {
    "identity": (-3, -3),
    "hflip": (0, 3),
    "vflip": (3, 0),
    "hvflip": (3, 3),
}
GEN_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _enclosed_interior(boundary_mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(boundary_mask)
    r0, c0, r1, c1 = _bbox(coords)
    sub = boundary_mask[r0 : r1 + 1, c0 : c1 + 1]
    sh, sw = sub.shape
    outside = np.zeros((sh, sw), dtype=bool)
    q = deque()

    for r in range(sh):
        for c in (0, sw - 1):
            if not sub[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(sw):
        for r in (0, sh - 1):
            if not sub[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < sh and 0 <= nc < sw and not sub[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    interior = ~sub & ~outside
    out = np.zeros_like(boundary_mask, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = interior
    return out


def _relative_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r) - 1, int(c) - 1) for r, c in np.argwhere(mask)]


def _transform_offsets(offsets: list[tuple[int, int]], branch: str) -> list[tuple[int, int]]:
    transformed = []
    for dr, dc in offsets:
        rr, cc = dr, dc
        if branch in ("hflip", "hvflip"):
            cc = -cc
        if branch in ("vflip", "hvflip"):
            rr = -rr
        transformed.append((rr, cc))
    return transformed


def _branch_from_delta(dr: int, dc: int) -> str:
    if dr == 0 and dc > 0:
        return "hflip"
    if dc == 0 and dr > 0:
        return "vflip"
    if dr > 0 and dc > 0:
        return "hvflip"
    if dr < 0 and dc < 0:
        return "identity"
    raise AssertionError("anchor must be to the right, below, lower-right, or upper-left of the source")


def _valid_source_patterns() -> list[np.ndarray]:
    patterns = []
    for bits in range(1 << 9):
        if bits.bit_count() < 3 or bits.bit_count() > 6:
            continue
        if not (bits & (1 << 4)):
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        if not mask[0].any() or not mask[2].any() or not mask[:, 0].any() or not mask[:, 2].any():
            continue
        if len(_components(mask, connectivity=4)) != 1:
            continue
        variants = {
            mask.tobytes(),
            np.fliplr(mask).tobytes(),
            np.flipud(mask).tobytes(),
            np.rot90(mask, 2).tobytes(),
        }
        if len(variants) != 4:
            continue
        patterns.append(mask)
    assert patterns
    return patterns


SOURCE_PATTERNS = _valid_source_patterns()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the source stamp to every anchor inside each enclosed region, choosing the mirror transform from the anchor's direction.

    Input structure:
    - The background is black and several large monochrome closed boundary loops each enclose one independent region.
    - Inside every region, exactly one small asymmetric connected source stamp and several isolated one-cell anchors share a third color distinct from the boundary.
    - The source stamp has 3-6 cells in a bbox no larger than 3x3, while every anchor is a singleton placed in one of the informative directions around that source.

    Rule:
    - Detect every closed boundary and the cells strictly enclosed by it.
    - Within each enclosed region, separate the shared-color cells into one multi-cell component (the source stamp) and singleton components (the anchors).
    - Use the source stamp bbox center as the stamping center. For an anchor directly to the right, paste a horizontal mirror of the stamp centered on the anchor; for one directly below, paste a vertical mirror; for one lower-right, paste both mirrors; for one upper-left, paste the stamp unchanged.
    - Keep the original boundaries and source stamps, and only paint copies that stay inside the same enclosed region.

    Key insight:
    - The anchors encode two things at once: where the copied stamp should appear and which mirror transformation to apply, so the solver must first distinguish source versus anchor roles before reading direction as flip information.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)
    out = g.copy()

    boundaries: list[tuple[int, np.ndarray]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color, connectivity=4):
            comp_mask = np.zeros((h, w), dtype=bool)
            comp_mask[comp[:, 0], comp[:, 1]] = True
            interior = _enclosed_interior(comp_mask)
            if interior.any():
                boundaries.append((color, interior))

    assert boundaries

    for boundary_color, interior in boundaries:
        inside_values = {int(v) for v in g[interior] if int(v) != bg}
        inside_values.discard(boundary_color)
        assert len(inside_values) == 1
        stamp_color = inside_values.pop()
        content_mask = interior & (g == stamp_color)
        parts = _components(content_mask, connectivity=8)
        source_parts = [part for part in parts if len(part) > 1]
        anchor_parts = [part for part in parts if len(part) == 1]

        assert len(source_parts) == 1
        assert anchor_parts

        source = source_parts[0]
        r0, c0, r1, c1 = _bbox(source)
        sh = r1 - r0 + 1
        sw = c1 - c0 + 1
        assert sh <= 3 and sw <= 3 and sh % 2 == 1 and sw % 2 == 1
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        assert g[center_r, center_c] == stamp_color

        source_offsets = [(int(r) - center_r, int(c) - center_c) for r, c in source]
        for anchor in anchor_parts:
            ar, ac = map(int, anchor[0])
            branch = _branch_from_delta(ar - center_r, ac - center_c)
            for dr, dc in _transform_offsets(source_offsets, branch):
                rr = ar + dr
                cc = ac + dc
                assert 0 <= rr < h and 0 <= cc < w and interior[rr, cc]
                out[rr, cc] = stamp_color

    return out


def _paint_region(
    patch: np.ndarray,
    pattern: np.ndarray,
    stamp_color: int,
    source_center: tuple[int, int],
    branches: list[str],
) -> np.ndarray:
    rel = _relative_coords(pattern)
    sr, sc = source_center
    for dr, dc in rel:
        patch[sr + dr, sc + dc] = stamp_color
    for branch in branches:
        off_r, off_c = BRANCH_OFFSETS[branch]
        patch[sr + off_r, sc + off_c] = stamp_color
    return patch


def _paint_region_output(
    patch: np.ndarray,
    pattern: np.ndarray,
    stamp_color: int,
    source_center: tuple[int, int],
    branches: list[str],
) -> np.ndarray:
    rel = _relative_coords(pattern)
    sr, sc = source_center
    for branch in branches:
        off_r, off_c = BRANCH_OFFSETS[branch]
        ar = sr + off_r
        ac = sc + off_c
        for dr, dc in _transform_offsets(rel, branch):
            patch[ar + dr, ac + dc] = stamp_color
    return patch


def _build_region(boundary_color: int, stamp_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    outer_h = random.randint(11, 12)
    outer_w = random.randint(11, 12)
    interior_h = outer_h - 2
    interior_w = outer_w - 2
    branches = random.sample(BRANCHES, random.randint(2, 4))
    pattern = random.choice(SOURCE_PATTERNS)

    row_min = 1
    row_max = interior_h - 2
    col_min = 1
    col_max = interior_w - 2

    if "identity" in branches:
        row_min = max(row_min, 4)
        col_min = max(col_min, 4)
    if "hflip" in branches:
        col_max = min(col_max, interior_w - 5)
    if "vflip" in branches:
        row_max = min(row_max, interior_h - 5)
    if "hvflip" in branches:
        row_max = min(row_max, interior_h - 5)
        col_max = min(col_max, interior_w - 5)

    if row_min > row_max or col_min > col_max:
        return None

    src_r = random.randint(row_min, row_max) + 1
    src_c = random.randint(col_min, col_max) + 1
    source_center = (src_r, src_c)

    inp = np.full((outer_h, outer_w), BG, dtype=int)
    inp[0, :] = boundary_color
    inp[-1, :] = boundary_color
    inp[:, 0] = boundary_color
    inp[:, -1] = boundary_color
    _paint_region(inp, pattern, stamp_color, source_center, branches)

    out = inp.copy()
    _paint_region_output(out, pattern, stamp_color, source_center, branches)
    return inp, out


def _slot_boxes(height: int, width: int) -> list[tuple[int, int, int, int]]:
    mid_r = height // 2
    mid_c = width // 2
    return [
        (0, mid_r, 0, mid_c),
        (0, mid_r, mid_c, width),
        (mid_r, height, 0, mid_c),
        (mid_r, height, mid_c, width),
    ]


def _sample_patch_position(
    canvas_shape: tuple[int, int], patch_shape: tuple[int, int], box: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    r0, r1, c0, c1 = box
    ph, pw = patch_shape
    top_margin = 1 if r0 == 0 else 0
    bottom_margin = 1 if r1 == canvas_shape[0] else 0
    left_margin = 1 if c0 == 0 else 0
    right_margin = 1 if c1 == canvas_shape[1] else 0

    min_r = r0 + top_margin
    max_r = r1 - ph - bottom_margin
    min_c = c0 + left_margin
    max_c = c1 - pw - right_margin
    if min_r > max_r or min_c > max_c:
        return None

    return random.randint(min_r, max_r), random.randint(min_c, max_c)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(28, 30)
    width = random.randint(28, 30)
    region_count = random.randint(2, 3)
    colors = random.sample(PALETTE, 2 * region_count)
    boundary_colors = colors[:region_count]
    stamp_colors = colors[region_count:]

    regions = []
    for boundary_color, stamp_color in zip(boundary_colors, stamp_colors):
        region = _build_region(boundary_color, stamp_color)
        if region is None:
            return None
        regions.append(region)

    boxes = _slot_boxes(height, width)
    chosen_boxes = random.sample(boxes, region_count)
    order = list(range(region_count))
    random.shuffle(order)

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    for idx, box in zip(order, chosen_boxes):
        region_inp, region_out = regions[idx]
        pos = _sample_patch_position(inp.shape, region_inp.shape, box)
        if pos is None:
            return None
        rr, cc = pos
        ph, pw = region_inp.shape
        inp[rr : rr + ph, cc : cc + pw] = region_inp
        out[rr : rr + ph, cc : cc + pw] = region_out

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
