from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

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
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            boundary[r, c] = True
            continue
        for dr, dc in DIR4:
            if not mask[r + dr, c + dc]:
                boundary[r, c] = True
                break
    return boundary


def _extract_side_objects(side: np.ndarray) -> list[dict]:
    objects = []
    colors = [int(v) for v in np.unique(side) if int(v) not in (BLACK, GREY)]
    for color in sorted(colors):
        comps = _components(side == color)
        assert len(comps) == 1, "each visible color should form exactly one shape per side"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        patch = side[r0 : r1 + 1, c0 : c1 + 1].copy()
        objects.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "patch": patch,
                "mask": patch == color,
            }
        )
    return objects


def _stripe_orientation(striped_patch: np.ndarray, color: int, solid_mask: np.ndarray) -> str:
    boundary = _boundary_mask(solid_mask)
    interior = solid_mask & (~boundary)
    assert interior.any(), "shape needs interior cells to encode stripes"

    row_mixed = 0
    row_uniform = 0
    for r in range(interior.shape[0]):
        cols = np.flatnonzero(interior[r])
        if cols.size == 0:
            continue
        vals = striped_patch[r, cols] == color
        if np.all(vals == vals[0]):
            row_uniform += 1
        else:
            row_mixed += 1

    col_mixed = 0
    col_uniform = 0
    for c in range(interior.shape[1]):
        rows = np.flatnonzero(interior[:, c])
        if rows.size == 0:
            continue
        vals = striped_patch[rows, c] == color
        if np.all(vals == vals[0]):
            col_uniform += 1
        else:
            col_mixed += 1

    if row_mixed < col_mixed:
        return "horizontal"
    if col_mixed < row_mixed:
        return "vertical"
    if row_uniform > col_uniform:
        return "horizontal"
    if col_uniform > row_uniform:
        return "vertical"
    raise AssertionError("stripe orientation is ambiguous")


def _stretch_patch(color: int, solid_mask: np.ndarray, stripe_orientation: str) -> np.ndarray:
    patch = np.zeros(solid_mask.shape, dtype=int)
    patch[solid_mask] = color
    if stripe_orientation == "horizontal":
        return np.repeat(patch, 2, axis=0)
    if stripe_orientation == "vertical":
        return np.repeat(patch, 2, axis=1)
    raise AssertionError("unknown stripe orientation")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each left template to its striped copy and stretch the template in the stripe-indicated direction.

    Input structure:
    - The background is black.
    - One full-height grey vertical line splits the grid into a left half and a right half.
    - The left half contains several small solid monochrome shapes, each using a distinct visible color.
    - The right half contains one same-color counterpart for each left shape, with the same outline but a striped interior made from that color and black.

    Rule:
    - Pair left and right shapes by their shared color, using the left shape as the solid template.
    - Read the stripe direction on the right: horizontal stripes mean the left template must be stretched vertically, while vertical stripes mean it must be stretched horizontally.
    - Build the output from the stretched left templates only, stacked from top to bottom in the same order as the left-half shapes and separated by one black row.

    Key insight:
    - The right-half shapes are instructions rather than output content: their stripe orientation encodes which axis the matching left template should be doubled along.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(sep_cols) == 1, "expected one grey separator column"
    sep = int(sep_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    left_objects = _extract_side_objects(left)
    right_objects = _extract_side_objects(right)

    assert left_objects and len(left_objects) == len(right_objects)
    right_by_color = {obj["color"]: obj for obj in right_objects}
    assert len(right_by_color) == len(right_objects)

    left_objects.sort(key=lambda obj: (obj["r0"], obj["c0"]))

    stretched = []
    for obj in left_objects:
        color = int(obj["color"])
        assert color in right_by_color, "every left shape must have one striped partner"
        partner = right_by_color[color]
        solid_mask = obj["mask"]
        assert partner["patch"].shape == solid_mask.shape, "paired shapes must share the same bbox"
        orientation = _stripe_orientation(partner["patch"], color, solid_mask)
        stretched.append(_stretch_patch(color, solid_mask, orientation))

    out_h = sum(p.shape[0] for p in stretched) + len(stretched) - 1
    out_w = max(p.shape[1] for p in stretched)
    out = np.zeros((out_h, out_w), dtype=int)

    r = 0
    for patch in stretched:
        h, w = patch.shape
        out[r : r + h, :w] = patch
        r += h + 1

    return out


def _random_mask() -> np.ndarray:
    h = random.randint(4, 5)
    w = random.randint(4, 5)
    mask = np.ones((h, w), dtype=bool)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    remove_count = random.randint(0, len(corners))
    for r, c in random.sample(corners, remove_count):
        mask[r, c] = False
    return mask


def _solid_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _striped_patch(mask: np.ndarray, color: int, orientation: str, phase: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    boundary = _boundary_mask(mask)
    interior = mask & (~boundary)
    patch[boundary] = color

    for r, c in np.argwhere(interior):
        if orientation == "horizontal":
            if (r + phase) % 2 == 0:
                patch[r, c] = color
        else:
            if (c + phase) % 2 == 0:
                patch[r, c] = color
    return patch


def _stack_output(patches: list[np.ndarray]) -> np.ndarray:
    out_h = sum(p.shape[0] for p in patches) + len(patches) - 1
    out_w = max(p.shape[1] for p in patches)
    out = np.zeros((out_h, out_w), dtype=int)
    r = 0
    for patch in patches:
        h, w = patch.shape
        out[r : r + h, :w] = patch
        r += h + 1
    return out


def _layout_stack(patches: list[np.ndarray], region_w: int) -> tuple[list[tuple[int, int]], int]:
    positions: list[tuple[int, int]] = []
    r = random.randint(1, 3)
    for i, patch in enumerate(patches):
        h, w = patch.shape
        c = random.randint(0, region_w - w)
        positions.append((r, c))
        r += h
        if i != len(patches) - 1:
            r += random.randint(1, 3)
    total_h = r + random.randint(1, 3)
    return positions, total_h


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _build_expected(masks: list[np.ndarray], colors: list[int], directions: list[str]) -> np.ndarray:
    patches = [_stretch_patch(color, mask, direction) for mask, color, direction in zip(masks, colors, directions)]
    return _stack_output(patches)


def _sample_case() -> np.ndarray | None:
    n_shapes = random.randint(2, 4)
    colors = random.sample(PALETTE, n_shapes)

    masks: list[np.ndarray] = []
    seen_masks: set[tuple[tuple[int, int], bytes]] = set()
    while len(masks) < n_shapes:
        mask = _random_mask()
        key = (mask.shape, mask.tobytes())
        if key in seen_masks:
            continue
        seen_masks.add(key)
        masks.append(mask)

    directions = [random.choice(("horizontal", "vertical")) for _ in range(n_shapes)]
    phases = [random.randint(0, 1) for _ in range(n_shapes)]

    expected = _build_expected(masks, colors, directions)
    if max(expected.shape) > 30:
        return None

    left_patches = [_solid_patch(mask, color) for mask, color in zip(masks, colors)]
    right_patches = [
        _striped_patch(mask, color, direction, phase)
        for mask, color, direction, phase in zip(masks, colors, directions, phases)
    ]

    left_region_w = max(p.shape[1] for p in left_patches) + random.randint(2, 7)
    right_region_w = max(p.shape[1] for p in right_patches) + random.randint(2, 7)
    total_w = left_region_w + 1 + right_region_w
    if total_w > 30:
        return None

    left_positions, left_h = _layout_stack(left_patches, left_region_w)

    right_order = list(range(n_shapes))
    random.shuffle(right_order)
    ordered_right = [right_patches[i] for i in right_order]
    right_positions, right_h = _layout_stack(ordered_right, right_region_w)

    min_h = max(left_h, right_h)
    if min_h > 30:
        return None
    total_h = min_h + random.randint(0, min(4, 30 - min_h))

    grid = np.zeros((total_h, total_w), dtype=int)
    grid[:, left_region_w] = GREY

    for patch, (r0, c0) in zip(left_patches, left_positions):
        _paste(grid, patch, r0, c0)

    for patch, (r0, c0) in zip(ordered_right, right_positions):
        _paste(grid, patch, r0, left_region_w + 1 + c0)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_case()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
