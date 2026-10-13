from __future__ import annotations

from collections import deque

import numpy as np


FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
EIGHT_NEIGHBORS = FOUR_NEIGHBORS + ((1, 1), (1, -1), (-1, 1), (-1, -1))

BG_COLOR = 0
CONTOUR_COLOR = 8
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 9)
DISTRACTOR_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 300
SHAPE_ATTEMPTS = 200

TRANSFORM_NAMES = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_lr_rot90",
    "flip_lr_rot180",
    "flip_lr_rot270",
)
NON_IDENTITY_TRANSFORMS = TRANSFORM_NAMES[1:]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return np.array(arr, copy=True)
    if name == "rot90":
        return np.rot90(arr, 1).copy()
    if name == "rot180":
        return np.rot90(arr, 2).copy()
    if name == "rot270":
        return np.rot90(arr, 3).copy()
    if name == "flip_lr":
        return np.fliplr(arr).copy()
    if name == "flip_lr_rot90":
        return np.rot90(np.fliplr(arr), 1).copy()
    if name == "flip_lr_rot180":
        return np.rot90(np.fliplr(arr), 2).copy()
    if name == "flip_lr_rot270":
        return np.rot90(np.fliplr(arr), 3).copy()
    raise AssertionError(f"unknown transform: {name}")


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    neighbors = EIGHT_NEIGHBORS if connectivity == 8 else FOUR_NEIGHBORS
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q = deque([(rr, cc)])
        coords: list[tuple[int, int]] = []
        while q:
            cr, dc = q.popleft()
            coords.append((int(cr), int(dc)))
            for dr, dd in neighbors:
                nr = int(cr + dr)
                nc = int(dc + dd)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _local_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _enclosed_cells(contour_mask: np.ndarray) -> np.ndarray:
    h, w = contour_mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if not contour_mask[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for dr, dc in FOUR_NEIGHBORS:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and not contour_mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return (~contour_mask) & (~exterior)


def _matching_transforms(src_mask: np.ndarray, dst_mask: np.ndarray) -> list[str]:
    matches: list[str] = []
    for name in TRANSFORM_NAMES:
        moved = _apply_transform(src_mask, name)
        if np.array_equal(moved, dst_mask):
            matches.append(name)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    candidates: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for coords in _components(g == color, connectivity=8):
            bbox, mask = _local_mask(coords)
            enclosed = _enclosed_cells(mask)
            if not bool(enclosed.any()):
                continue
            r0, c0, r1, c1 = bbox
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            pattern_mask = enclosed & (patch != bg) & (patch != color)
            candidates.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "mask": mask,
                    "pattern_mask": pattern_mask,
                    "pattern_count": int(pattern_mask.sum()),
                }
            )

    chosen: tuple[dict[str, object], dict[str, object], str] | None = None
    for color in sorted({int(item["color"]) for item in candidates}):
        group = [item for item in candidates if int(item["color"]) == color]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                first = group[i]
                second = group[j]
                first_has_pattern = int(first["pattern_count"]) > 0
                second_has_pattern = int(second["pattern_count"]) > 0
                if first_has_pattern == second_has_pattern:
                    continue
                template = first if first_has_pattern else second
                target = second if first_has_pattern else first
                matches = _matching_transforms(
                    np.asarray(template["mask"], dtype=bool),
                    np.asarray(target["mask"], dtype=bool),
                )
                if len(matches) != 1:
                    continue
                if chosen is not None:
                    raise AssertionError("expected a unique template-target pair")
                chosen = (template, target, matches[0])

    assert chosen is not None, "failed to identify template and target"
    template, target, transform_name = chosen

    tr0, tc0, tr1, tc1 = template["bbox"]
    template_patch = g[int(tr0) : int(tr1) + 1, int(tc0) : int(tc1) + 1]
    pattern = np.full(template_patch.shape, bg, dtype=int)
    pattern_mask = np.asarray(template["pattern_mask"], dtype=bool)
    pattern[pattern_mask] = template_patch[pattern_mask]

    moved_pattern = _apply_transform(pattern, transform_name)
    rr0, cc0, rr1, cc1 = target["bbox"]
    assert moved_pattern.shape == (int(rr1 - rr0 + 1), int(cc1 - cc0 + 1))

    out = g.copy()
    target_patch = out[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1]
    fill_mask = moved_pattern != bg
    target_patch[fill_mask] = moved_pattern[fill_mask]
    out[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1] = target_patch
    return out


def _erode4(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _symmetry_count(mask: np.ndarray) -> int:
    return sum(int(np.array_equal(_apply_transform(mask, name), mask)) for name in TRANSFORM_NAMES)


def _grow_pattern_mask(rng: np.random.Generator, interior: np.ndarray, contour: np.ndarray, size: int) -> np.ndarray | None:
    cells = np.argwhere(interior)
    if len(cells) < size:
        return None

    touching = []
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        if any(
            0 <= rr + dr < contour.shape[0]
            and 0 <= cc + dc < contour.shape[1]
            and contour[rr + dr, cc + dc]
            for dr, dc in EIGHT_NEIGHBORS
        ):
            touching.append((rr, cc))
    if not touching:
        return None

    chosen: set[tuple[int, int]] = {touching[int(rng.integers(0, len(touching)))]}
    while len(chosen) < size:
        frontier: list[tuple[int, int]] = []
        for r, c in chosen:
            for dr, dc in EIGHT_NEIGHBORS:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < interior.shape[0] and 0 <= nc < interior.shape[1] and interior[nr, nc] and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        frontier = list(dict.fromkeys(frontier))
        chosen.add(frontier[int(rng.integers(0, len(frontier)))])

    mask = np.zeros_like(interior, dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _generate_contour_shape(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_ATTEMPTS):
        base_h = int(rng.integers(5, 8))
        base_w = int(rng.integers(5, 8))
        canvas = np.zeros((base_h + 6, base_w + 6), dtype=bool)
        r0 = 3
        c0 = 3
        canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

        tab_count = int(rng.integers(2, 5))
        for _ in range(tab_count):
            side = int(rng.integers(0, 4))
            if side in (0, 1):
                max_len = max(2, base_w - 1)
                length = int(rng.integers(1, max_len + 1))
                thickness = int(rng.integers(1, 3))
                start = int(rng.integers(0, base_w - length + 1))
                if side == 0:
                    canvas[r0 - thickness : r0, c0 + start : c0 + start + length] = True
                else:
                    canvas[r0 + base_h : r0 + base_h + thickness, c0 + start : c0 + start + length] = True
            else:
                max_len = max(2, base_h - 1)
                length = int(rng.integers(1, max_len + 1))
                thickness = int(rng.integers(1, 3))
                start = int(rng.integers(0, base_h - length + 1))
                if side == 2:
                    canvas[r0 + start : r0 + start + length, c0 - thickness : c0] = True
                else:
                    canvas[r0 + start : r0 + start + length, c0 + base_w : c0 + base_w + thickness] = True

        outer = _trim(canvas)
        interior = _erode4(outer)
        if int(interior.sum()) < 8:
            continue
        contour = outer & (~interior)
        if len(_components(contour, connectivity=8)) != 1:
            continue
        if len(_components(interior, connectivity=4)) != 1:
            continue
        if _symmetry_count(contour) != 1:
            continue
        return contour, interior
    return None


def _generate_pattern_patch(
    rng: np.random.Generator,
    contour: np.ndarray,
    interior: np.ndarray,
) -> np.ndarray | None:
    max_cells = min(6, int(interior.sum()))
    if max_cells < 3:
        return None
    size = int(rng.integers(3, max_cells + 1))
    pattern_mask = _grow_pattern_mask(rng, interior, contour, size)
    if pattern_mask is None:
        return None

    color_count = 2 if size == 3 else int(rng.integers(2, min(3, size) + 1))
    colors = [int(v) for v in rng.choice(np.array(PATTERN_COLORS, dtype=int), size=color_count, replace=False)]
    coords = np.argwhere(pattern_mask)
    coords = coords[rng.permutation(len(coords))]

    patch = np.full(contour.shape, BG_COLOR, dtype=int)
    for idx, (r, c) in enumerate(coords):
        if idx < color_count:
            color = colors[idx]
        else:
            color = colors[int(rng.integers(0, color_count))]
        patch[int(r), int(c)] = int(color)
    return patch


def _rects_separated(
    r0: int,
    c0: int,
    h: int,
    w: int,
    others: list[tuple[int, int, int, int]],
    gap: int = 1,
) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for or0, oc0, oh, ow in others:
        or1 = or0 + oh - 1
        oc1 = oc0 + ow - 1
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _place_boxes(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    rects: list[tuple[int, int, int, int]] = []
    pos_by_idx: list[tuple[int, int] | None] = [None] * len(sizes)
    for idx in order:
        h, w = sizes[idx]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(0, grid_h - h + 1))
            c0 = int(rng.integers(0, grid_w - w + 1))
            if _rects_separated(r0, c0, h, w, rects, gap=1):
                rects.append((r0, c0, h, w))
                pos_by_idx[idx] = (r0, c0)
                placed = True
                break
        if not placed:
            return None
    return [pos for pos in pos_by_idx if pos is not None]


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG_COLOR
    view = grid[top : top + h, left : left + w]
    view[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    shape = _generate_contour_shape(rng)
    if shape is None:
        return None
    canonical_contour, canonical_interior = shape
    canonical_pattern = _generate_pattern_patch(rng, canonical_contour, canonical_interior)
    if canonical_pattern is None:
        return None

    template_orientation = TRANSFORM_NAMES[int(rng.integers(0, len(TRANSFORM_NAMES)))]
    relation_transform = NON_IDENTITY_TRANSFORMS[int(rng.integers(0, len(NON_IDENTITY_TRANSFORMS)))]

    template_contour = _apply_transform(canonical_contour, template_orientation)
    template_pattern = _apply_transform(canonical_pattern, template_orientation)
    target_contour = _apply_transform(template_contour, relation_transform)
    target_pattern = _apply_transform(template_pattern, relation_transform)

    template_patch = np.full(template_contour.shape, BG_COLOR, dtype=int)
    template_patch[template_contour] = CONTOUR_COLOR
    template_patch[template_pattern != BG_COLOR] = template_pattern[template_pattern != BG_COLOR]

    target_patch = np.full(target_contour.shape, BG_COLOR, dtype=int)
    target_patch[target_contour] = CONTOUR_COLOR

    patches = [template_patch, target_patch]

    add_distractor = bool(rng.integers(0, 2))
    if add_distractor:
        d_h = int(rng.integers(3, 6))
        d_w = int(rng.integers(3, 6))
        distractor_color_choices = [c for c in DISTRACTOR_COLORS if c not in set(int(v) for v in np.unique(template_pattern) if int(v) != BG_COLOR)]
        if not distractor_color_choices:
            return None
        distractor_color = int(distractor_color_choices[int(rng.integers(0, len(distractor_color_choices)))])
        distractor = np.full((d_h, d_w), BG_COLOR, dtype=int)
        distractor[0, :] = distractor_color
        distractor[-1, :] = distractor_color
        distractor[:, 0] = distractor_color
        distractor[:, -1] = distractor_color
        patches.append(distractor)

    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    min_h = max(16, max_h + 4)
    min_w = max(18, max_w + 4)
    if min_h > 22 or min_w > 24:
        return None

    grid_h = int(rng.integers(min_h, 23))
    grid_w = int(rng.integers(min_w, 25))
    positions = _place_boxes(rng, grid_h, grid_w, [p.shape for p in patches])
    if positions is None:
        return None

    inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        _stamp_patch(inp, patch, int(r0), int(c0))

    expected = inp.copy()
    rr0, cc0 = positions[1]
    moved_mask = target_pattern != BG_COLOR
    expected[rr0 : rr0 + target_pattern.shape[0], cc0 : cc0 + target_pattern.shape[1]][moved_mask] = target_pattern[moved_mask]

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
