from __future__ import annotations

from collections import Counter, deque

import numpy as np


BACKGROUND = 0
GRAY = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
OFFSETS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
OFFSETS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
MAX_GENERATION_ATTEMPTS = 400
MAX_MASK_ATTEMPTS = 120
MAX_PLACEMENT_ATTEMPTS = 120


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    offsets = OFFSETS4 if int(connectivity) == 4 else OFFSETS8
    components: list[np.ndarray] = []

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
                for dr, dc in offsets:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _has_hole(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque([(0, 0)])
    seen[0, 0] = True

    while q:
        r, c = q.popleft()
        for dr, dc in OFFSETS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1]:
                if not padded[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))

    return bool(((~padded) & (~seen)).any())


def _interior_boundary_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    interior = np.zeros_like(mask, dtype=bool)

    for r, c in np.argwhere(mask):
        if padded[r : r + 3, c : c + 3].all():
            interior[r, c] = True

    boundary = mask & ~interior
    return interior, boundary


def _winner_from_counts(counts: Counter[int]) -> int:
    if not counts:
        return BACKGROUND
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    return int(winners[0]) if len(winners) == 1 else GRAY


def _count_values(values: np.ndarray) -> Counter[int]:
    return Counter(int(v) for v in np.asarray(values).ravel())


def _component_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    comp_mask = np.zeros(grid.shape, dtype=bool)
    comp_mask[coords[:, 0], coords[:, 1]] = True
    interior, boundary = _interior_boundary_masks(comp_mask)
    r0, c0, r1, c1 = _bbox(coords)
    local_mask = comp_mask[r0 : r1 + 1, c0 : c1 + 1]
    local_grid = grid[r0 : r1 + 1, c0 : c1 + 1]

    outer_frame = np.zeros_like(local_mask, dtype=bool)
    outer_frame[0, :] = True
    outer_frame[-1, :] = True
    outer_frame[:, 0] = True
    outer_frame[:, -1] = True
    bbox_ring = outer_frame & local_mask

    colors = sorted(set(int(v) for v in grid[comp_mask]) - {BACKGROUND})
    boundary_counts = _count_values(grid[boundary])
    total_counts = _count_values(grid[comp_mask])
    bbox_counts = _count_values(local_grid[bbox_ring])

    return {
        "colors": colors,
        "area": int(comp_mask.sum()),
        "boundary_size": int(boundary.sum()),
        "interior_size": int(interior.sum()),
        "vote": _winner_from_counts(boundary_counts),
        "global_vote": _winner_from_counts(total_counts),
        "bbox_vote": _winner_from_counts(bbox_counts),
        "boundary_counts": boundary_counts,
        "is_rect": bool(local_mask.all()),
        "has_hole": _has_hole(local_mask),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each bicolor blob to the majority color among its boundary pixels.

    Input structure:
    - Black background with several disconnected solid blobs.
    - Each blob is a hole-free foreground component made from two non-black colors.
    - Thick blobs may contain interior pixels whose full 3x3 neighborhoods stay inside the blob.

    Rule:
    - Treat each 8-connected foreground component as one blob.
    - Within a blob, a pixel is interior iff all eight neighbors also belong to that same blob; the rest are boundary pixels.
    - Count only the boundary pixels' colors. A unique boundary majority recolors the whole blob to that color.
    - If the boundary vote is tied, recolor the whole blob to gray.

    Key insight:
    - The decisive color comes from the blob's outer ring, not from the blob-wide totals, so the solver must peel off 3x3-supported interiors before counting.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for coords in _components_mask(g != BACKGROUND, connectivity=8):
        comp_mask = np.zeros(g.shape, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        _interior, boundary = _interior_boundary_masks(comp_mask)
        winner = _winner_from_counts(_count_values(g[boundary]))
        out[comp_mask] = winner

    return out


def _make_rectangle(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(4, 9))
    w = int(rng.integers(4, 9))
    return np.ones((h, w), dtype=bool)


def _make_notched_rectangle(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(6, 10))
    w = int(rng.integers(6, 10))
    mask = np.ones((h, w), dtype=bool)
    notch_h = int(rng.integers(1, min(3, h - 3) + 1))
    notch_w = int(rng.integers(1, min(3, w - 3) + 1))
    corner = int(rng.integers(0, 4))

    if corner == 0:
        mask[:notch_h, :notch_w] = False
    elif corner == 1:
        mask[:notch_h, -notch_w:] = False
    elif corner == 2:
        mask[-notch_h:, :notch_w] = False
    else:
        mask[-notch_h:, -notch_w:] = False

    return mask


def _make_blob_mask(
    rng: np.random.Generator,
    *,
    require_irregular: bool = False,
    require_even_boundary: bool = False,
    require_inner_boundary: bool = False,
) -> np.ndarray | None:
    for _ in range(MAX_MASK_ATTEMPTS):
        if require_irregular or rng.random() < 0.55:
            mask = _make_notched_rectangle(rng)
        else:
            mask = _make_rectangle(rng)

        interior, boundary = _interior_boundary_masks(mask)
        outer_frame = np.zeros_like(mask, dtype=bool)
        outer_frame[0, :] = True
        outer_frame[-1, :] = True
        outer_frame[:, 0] = True
        outer_frame[:, -1] = True
        inner_boundary = boundary & ~outer_frame

        if int(interior.sum()) < 2:
            continue
        if require_irregular and bool(mask.all()):
            continue
        if require_even_boundary and int(boundary.sum()) % 2 != 0:
            continue
        if require_inner_boundary and not bool(inner_boundary.any()):
            continue
        if _has_hole(mask):
            continue
        return mask
    return None


def _choose_two_colors(rng: np.random.Generator) -> tuple[int, int]:
    colors = rng.choice(np.array(INPUT_COLORS, dtype=int), size=2, replace=False)
    return int(colors[0]), int(colors[1])


def _build_special_blob(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(MAX_MASK_ATTEMPTS):
        mask = _make_blob_mask(rng, require_irregular=True, require_inner_boundary=True)
        if mask is None:
            continue

        interior, boundary = _interior_boundary_masks(mask)
        outer_frame = np.zeros_like(mask, dtype=bool)
        outer_frame[0, :] = True
        outer_frame[-1, :] = True
        outer_frame[:, 0] = True
        outer_frame[:, -1] = True
        outer = boundary & outer_frame
        inner = boundary & ~outer_frame
        if not outer.any() or not inner.any():
            continue

        bbox_color, true_border_color = _choose_two_colors(rng)
        if int(rng.integers(0, 2)):
            bbox_color, true_border_color = true_border_color, bbox_color

        patch = np.zeros(mask.shape, dtype=int)
        outer_coords = np.argwhere(outer)
        rng.shuffle(outer_coords)
        bbox_majority = len(outer_coords) // 2 + 1
        for r, c in outer_coords[:bbox_majority]:
            patch[r, c] = bbox_color
        for r, c in outer_coords[bbox_majority:]:
            patch[r, c] = true_border_color
        patch[inner] = true_border_color
        patch[interior] = bbox_color

        info = _component_info(patch, np.argwhere(patch != BACKGROUND))
        if (
            len(info["colors"]) == 2
            and len(info["boundary_counts"]) == 2
            and info["vote"] == true_border_color
            and info["bbox_vote"] == bbox_color
            and info["global_vote"] == bbox_color
        ):
            return patch

    return None


def _build_tie_blob(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(MAX_MASK_ATTEMPTS):
        mask = _make_blob_mask(
            rng,
            require_irregular=bool(int(rng.integers(0, 2))),
            require_even_boundary=True,
        )
        if mask is None:
            continue

        interior, boundary = _interior_boundary_masks(mask)
        color_a, color_b = _choose_two_colors(rng)
        fill_color = color_a if int(rng.integers(0, 2)) == 0 else color_b
        other_color = color_b if fill_color == color_a else color_a

        patch = np.zeros(mask.shape, dtype=int)
        boundary_coords = np.argwhere(boundary)
        rng.shuffle(boundary_coords)
        half = len(boundary_coords) // 2
        for r, c in boundary_coords[:half]:
            patch[r, c] = fill_color
        for r, c in boundary_coords[half:]:
            patch[r, c] = other_color
        patch[interior] = fill_color

        info = _component_info(patch, np.argwhere(patch != BACKGROUND))
        if (
            len(info["colors"]) == 2
            and len(info["boundary_counts"]) == 2
            and info["vote"] == GRAY
            and info["global_vote"] != GRAY
        ):
            return patch

    return None


def _build_normal_blob(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(MAX_MASK_ATTEMPTS):
        mask = _make_blob_mask(rng, require_irregular=bool(rng.random() < 0.45))
        if mask is None:
            continue

        interior, boundary = _interior_boundary_masks(mask)
        winner, loser = _choose_two_colors(rng)
        if int(rng.integers(0, 2)):
            winner, loser = loser, winner

        patch = np.zeros(mask.shape, dtype=int)
        boundary_coords = np.argwhere(boundary)
        rng.shuffle(boundary_coords)
        min_majority = len(boundary_coords) // 2 + 1
        if min_majority >= len(boundary_coords):
            continue
        majority_count = int(rng.integers(min_majority, len(boundary_coords)))
        for r, c in boundary_coords[:majority_count]:
            patch[r, c] = winner
        for r, c in boundary_coords[majority_count:]:
            patch[r, c] = loser
        patch[interior] = winner

        info = _component_info(patch, np.argwhere(patch != BACKGROUND))
        if (
            len(info["colors"]) == 2
            and len(info["boundary_counts"]) == 2
            and info["vote"] == winner
            and info["global_vote"] == winner
        ):
            return patch

    return None


def _build_reverse_blob(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(MAX_MASK_ATTEMPTS):
        mask = _make_blob_mask(rng, require_irregular=bool(rng.random() < 0.35))
        if mask is None:
            continue

        interior, boundary = _interior_boundary_masks(mask)
        border_winner, interior_winner = _choose_two_colors(rng)
        if int(rng.integers(0, 2)):
            border_winner, interior_winner = interior_winner, border_winner

        patch = np.zeros(mask.shape, dtype=int)
        boundary_coords = np.argwhere(boundary)
        rng.shuffle(boundary_coords)
        min_majority = len(boundary_coords) // 2 + 1
        if min_majority >= len(boundary_coords):
            continue
        majority_count = min_majority
        for r, c in boundary_coords[:majority_count]:
            patch[r, c] = border_winner
        for r, c in boundary_coords[majority_count:]:
            patch[r, c] = interior_winner
        patch[interior] = interior_winner

        info = _component_info(patch, np.argwhere(patch != BACKGROUND))
        if (
            len(info["colors"]) == 2
            and len(info["boundary_counts"]) == 2
            and info["vote"] == border_winner
            and info["global_vote"] == interior_winner
        ):
            return patch

    return None


def _can_place(occupied: np.ndarray, patch_mask: np.ndarray, top: int, left: int) -> bool:
    h, w = patch_mask.shape
    for r, c in np.argwhere(patch_mask):
        rr, cc = top + int(r), left + int(c)
        r0 = max(0, rr - 1)
        r1 = min(occupied.shape[0], rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(occupied.shape[1], cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, rng: np.random.Generator) -> bool:
    patch_mask = patch != BACKGROUND
    h, w = patch.shape
    if grid.shape[0] - h < 2 or grid.shape[1] - w < 2:
        return False

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        top = int(rng.integers(1, grid.shape[0] - h))
        left = int(rng.integers(1, grid.shape[1] - w))
        if not _can_place(occupied, patch_mask, top, left):
            continue
        window = grid[top : top + h, left : left + w]
        window[patch_mask] = patch[patch_mask]
        occupied[top : top + h, left : left + w] |= patch_mask
        return True
    return False


def _semantic_ok(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if GRAY in set(int(v) for v in np.unique(grid) if int(v) != BACKGROUND):
        return False

    components = _components_mask(grid != BACKGROUND, connectivity=8)
    if not (3 <= len(components) <= 5):
        return False

    infos = [_component_info(grid, coords) for coords in components]
    if any(len(info["colors"]) != 2 for info in infos):
        return False
    if any(len(info["boundary_counts"]) != 2 for info in infos):
        return False
    if any(info["interior_size"] == 0 for info in infos):
        return False
    if any(info["has_hole"] for info in infos):
        return False
    if not any(info["vote"] == GRAY for info in infos):
        return False
    if not any(info["vote"] != GRAY and info["vote"] == info["global_vote"] for info in infos):
        return False
    if not any(info["vote"] != info["global_vote"] and info["global_vote"] != GRAY for info in infos):
        return False
    if not any((not info["is_rect"]) and info["vote"] != info["bbox_vote"] for info in infos):
        return False

    out = main(grid)
    votes = {info["vote"] for info in infos}
    out_colors = {int(v) for v in np.unique(out) if int(v) != BACKGROUND}
    if np.array_equal(out, grid):
        return False
    if votes != out_colors:
        return False
    if len(votes) < 3 or GRAY not in votes:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    builders = [_build_special_blob, _build_tie_blob, _build_normal_blob]
    extra_count = int(rng.integers(0, 3))
    for _ in range(extra_count):
        builders.append(_build_reverse_blob if rng.random() < 0.45 else _build_normal_blob)

    patches: list[np.ndarray] = []
    for builder in builders:
        patch = builder(rng)
        if patch is None:
            return None
        patches.append(patch)

    patches.sort(key=lambda patch: int(np.count_nonzero(patch)), reverse=True)
    for patch in patches:
        if not _place_patch(grid, occupied, patch, rng):
            return None

    return grid if _semantic_ok(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid boundary-voting ARC input")
