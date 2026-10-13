from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _deep_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if all(mask[r + dr, c + dc] for dr, dc in DIRS4):
                out.append((r, c))
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        marker = np.argwhere(region == YELLOW)
        assert marker.shape == (1, 2), "each object must contain exactly one yellow marker"
        body_colors = sorted(set(map(int, region[region != BG].ravel())) - {YELLOW})
        assert len(body_colors) == 1, "each object must have one non-yellow body color"
        body = body_colors[0]
        mask = region != BG
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": mask.shape,
                "mask": mask,
                "body": body,
                "marker_rel": (int(marker[0, 0]), int(marker[0, 1])),
                "marker_abs": (int(r0 + marker[0, 0]), int(c0 + marker[0, 1])),
            }
        )
    return objects


def _scaled_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((k, k), dtype=int)).astype(bool)


def _pairwise_overlap_count(mask: np.ndarray, dr: int, dc: int) -> int:
    h, w = mask.shape
    keep = 0
    for r, c in np.argwhere(mask):
        sr = int(r) - dr
        sc = int(c) - dc
        if 0 <= sr < h and 0 <= sc < w and mask[sr, sc]:
            keep += 1
    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each small yellow-marked sprite with its large same-color counterpart and grey the large sprite where the aligned enlargement is blank.

    Input structure:
    - The background is black.
    - Every non-black 4-connected object contains exactly one yellow marker pixel and one other body color.
    - Each body color appears in exactly two objects: a small sprite and a larger same-orientation sprite of the same shape.
    - The large sprite's bbox dimensions are an integer multiple of the small sprite's bbox dimensions.

    Rule:
    - Group objects by body color, identify the smaller and larger member, and derive the integer scale factor from their bbox ratio.
    - Replace the small marker by body-colored occupancy, enlarge the small mask by that factor, and align the enlarged marker block's top-left corner to the large yellow marker.
    - Remove every small sprite entirely.
    - For each pixel of the large sprite, repaint it grey when the aligned enlarged mask is background or out of bounds there; otherwise keep the body color. Replace the large yellow marker with the body color.

    Key insight:
    - The yellow pixels are alignment anchors, and the visible output is the complement of the aligned enlarged small mask over the large object rather than a pasted copy of the small object's colored cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    by_color: dict[int, list[dict]] = {}
    for obj in _extract_objects(g):
        by_color.setdefault(int(obj["body"]), []).append(obj)

    assert by_color, "expected at least one colored object pair"
    out = np.zeros_like(g)

    for body, pair in by_color.items():
        assert len(pair) == 2, "each body color must appear in exactly two objects"
        pair = sorted(pair, key=lambda obj: obj["shape"][0] * obj["shape"][1])
        small, big = pair
        sh, sw = small["shape"]
        bh, bw = big["shape"]
        kh, rem_h = divmod(bh, sh)
        kw, rem_w = divmod(bw, sw)
        assert rem_h == 0 and rem_w == 0 and kh == kw and kh > 1, "expected one integer scaling factor"
        k = kh

        scaled_small = _scaled_mask(small["mask"], k)
        assert scaled_small.shape == big["shape"], "scaled small bbox must match large bbox"
        assert np.array_equal(scaled_small, big["mask"]), "paired objects must share the same scaled shape"

        anchor_r = small["marker_rel"][0] * k
        anchor_c = small["marker_rel"][1] * k
        top_r = big["marker_abs"][0] - anchor_r
        top_c = big["marker_abs"][1] - anchor_c

        br0, bc0, _br1, _bc1 = big["bbox"]
        for rr, cc in np.argwhere(big["mask"]):
            ar = br0 + int(rr)
            ac = bc0 + int(cc)
            sr = ar - top_r
            sc = ac - top_c
            covered = 0 <= sr < bh and 0 <= sc < bw and scaled_small[sr, sc]
            out[ar, ac] = body if covered else GREY

    return out


PALETTE = (1, 2, 3, 6, 7, 8, 9)
CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 500
PAIR_COUNT_CHOICES = (1, 2, 3)
K_CHOICES = (2, 3)
SMALL_DIM_MIN = 4
SMALL_DIM_MAX = 6
PLACEMENT_PAD = 1


def _is_connected(mask: np.ndarray) -> bool:
    true_count = int(mask.sum())
    if true_count == 0:
        return False
    return len(_components(mask)) == 1


def _random_small_mask() -> np.ndarray | None:
    h = random.randint(SMALL_DIM_MIN, SMALL_DIM_MAX)
    w = random.randint(SMALL_DIM_MIN, SMALL_DIM_MAX)
    mask = np.ones((h, w), dtype=bool)

    max_remove = min(5, h * w // 3)
    target_remove = random.randint(0, max_remove)
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)

    removed = 0
    for r, c in cells:
        if removed >= target_remove:
            break
        if int(mask.sum()) <= max(10, h + w):
            break
        trial = mask.copy()
        trial[r, c] = False
        if not trial[0, :].any() or not trial[-1, :].any() or not trial[:, 0].any() or not trial[:, -1].any():
            continue
        if not _is_connected(trial):
            continue
        mask = trial
        removed += 1

    if len(_deep_cells(mask)) < 2:
        return None
    return mask


def _choose_markers(small_mask: np.ndarray, k: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    small_candidates = _deep_cells(small_mask)
    big_mask = _scaled_mask(small_mask, k)
    big_candidates = _deep_cells(big_mask)
    big_area = int(big_mask.sum())
    random.shuffle(small_candidates)
    random.shuffle(big_candidates)

    for sr, sc in small_candidates:
        canonical = (sr * k, sc * k)
        valid_big: list[tuple[int, int]] = []
        for br, bc in big_candidates:
            if (br, bc) == canonical:
                continue
            keep = _pairwise_overlap_count(big_mask, br - canonical[0], bc - canonical[1])
            grey = big_area - keep
            if keep >= max(4, big_area // 4) and grey >= max(1, big_area // 10):
                valid_big.append((br, bc))
        if valid_big:
            return (sr, sc), random.choice(valid_big)
    return None


def _make_patch(mask: np.ndarray, color: int, marker: tuple[int, int]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    patch[marker] = YELLOW
    return patch


def _crop_with_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(3, int(r0)))
    left = random.randint(0, min(3, int(c0)))
    bottom = random.randint(0, min(3, int(grid.shape[0] - 1 - r1)))
    right = random.randint(0, min(3, int(grid.shape[1] - 1 - c1)))
    return grid[int(r0) - top : int(r1) + 1 + bottom, int(c0) - left : int(c1) + 1 + right]


def _free_positions(occupied: np.ndarray, ph: int, pw: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - PLACEMENT_PAD)
            cc0 = max(0, c0 - PLACEMENT_PAD)
            rr1 = min(h, r0 + ph + PLACEMENT_PAD)
            cc1 = min(w, c0 + pw + PLACEMENT_PAD)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    positions = _free_positions(occupied, patch.shape[0], patch.shape[1])
    if not positions:
        return False
    r0, c0 = random.choice(positions)
    nz = patch != BG
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][nz] = patch[nz]
    occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = True
    return True


def _build_pair(color: int) -> tuple[np.ndarray, np.ndarray] | None:
    small_mask = _random_small_mask()
    if small_mask is None:
        return None
    k = random.choice(K_CHOICES)
    markers = _choose_markers(small_mask, k)
    if markers is None:
        return None
    small_marker, big_marker = markers
    big_mask = _scaled_mask(small_mask, k)
    return _make_patch(big_mask, color, big_marker), _make_patch(small_mask, color, small_marker)


def _build_input() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(PALETTE, pair_count)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    total_rect_area = 0

    for color in colors:
        pair = _build_pair(color)
        if pair is None:
            return None
        big_patch, small_patch = pair
        total_rect_area += int(np.prod(big_patch.shape) + np.prod(small_patch.shape))
        pairs.append(pair)

    if total_rect_area > 520:
        return None

    grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    patches = [patch for pair in pairs for patch in pair]
    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    for patch in patches:
        if not _place_patch(grid, occupied, patch):
            return None

    cropped = _crop_with_margin(grid)
    try:
        out = main(cropped)
    except AssertionError:
        return None

    if np.any(out == YELLOW):
        return None
    if not np.any(out == GREY):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
