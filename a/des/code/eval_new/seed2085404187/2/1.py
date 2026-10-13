from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 400
GRID_MIN = 22
GRID_MAX = 30
EDGE_MARGIN = 5
OUTPUT_GAP = 1
EXTRA_CLEAR_MAX = 0


@dataclass(frozen=True)
class ContainerMeta:
    seed_r: int
    seed_c: int
    color: int
    direction: str
    thickness: int


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _full_rows_from_top(mask: np.ndarray) -> int:
    count = 0
    for r in range(mask.shape[0]):
        if np.all(mask[r, :]):
            count += 1
        else:
            break
    return count


def _full_rows_from_bottom(mask: np.ndarray) -> int:
    count = 0
    for r in range(mask.shape[0] - 1, -1, -1):
        if np.all(mask[r, :]):
            count += 1
        else:
            break
    return count


def _full_cols_from_left(mask: np.ndarray) -> int:
    count = 0
    for c in range(mask.shape[1]):
        if np.all(mask[:, c]):
            count += 1
        else:
            break
    return count


def _full_cols_from_right(mask: np.ndarray) -> int:
    count = 0
    for c in range(mask.shape[1] - 1, -1, -1):
        if np.all(mask[:, c]):
            count += 1
        else:
            break
    return count


def _container_specs(g: np.ndarray) -> list[ContainerMeta]:
    specs: list[ContainerMeta] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        wall = patch == GREY

        top_t = _full_rows_from_top(wall)
        bottom_t = _full_rows_from_bottom(wall)
        left_t = _full_cols_from_left(wall)
        right_t = _full_cols_from_right(wall)

        direction: str | None = None
        thickness = 0
        if top_t == 0 and min(bottom_t, left_t, right_t) > 0:
            direction = "up"
            thickness = min(left_t, right_t)
        elif bottom_t == 0 and min(top_t, left_t, right_t) > 0:
            direction = "down"
            thickness = min(left_t, right_t)
        elif left_t == 0 and min(top_t, bottom_t, right_t) > 0:
            direction = "left"
            thickness = min(top_t, bottom_t)
        elif right_t == 0 and min(top_t, bottom_t, left_t) > 0:
            direction = "right"
            thickness = min(top_t, bottom_t)
        else:
            raise AssertionError("expected each grey component to be a single open container")

        seed_coords = np.argwhere((patch != BG) & (patch != GREY))
        assert seed_coords.shape == (1, 2), "each container bbox must contain exactly one seed"
        sr = r0 + int(seed_coords[0, 0])
        sc = c0 + int(seed_coords[0, 1])
        specs.append(ContainerMeta(sr, sc, int(g[sr, sc]), direction, int(thickness)))
    return specs


def _apply_growth(grid: np.ndarray, specs: list[ContainerMeta]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    for spec in specs:
        dr, dc = DIRS[spec.direction]
        for step in range(1, spec.thickness + 1):
            rr = spec.seed_r + dr * step
            cc = spec.seed_c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if int(out[rr, cc]) != BG:
                break
            out[rr, cc] = spec.color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a short colored branch out of each open grey container using the seed color and the wall thickness.

    Input structure:
    - The background is black.
    - Several grey connected components form rectangular three-sided containers with one open side.
    - Each container has exactly one non-grey, non-black seed pixel inside its cavity.

    Rule:
    - For each grey container, identify the missing wall side to determine the opening direction.
    - Measure the wall thickness as the smaller thickness of the two walls perpendicular to that opening.
    - Starting from the seed, extend a 1-pixel-wide line in the opening direction for exactly that many cells.
    - Keep the grey container and the seed unchanged, and stop early if the branch would hit a non-background cell or the grid boundary.

    Key insight:
    - The opening tells you where the branch grows, but the encoded branch length comes from the container wall thickness rather than the cavity size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _apply_growth(g, _container_specs(g))


def _rotate_point(r: int, c: int, h: int, w: int, k: int) -> tuple[int, int]:
    if k % 4 == 0:
        return r, c
    if k % 4 == 1:
        return w - 1 - c, r
    if k % 4 == 2:
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _rotate_dir(direction: str, k: int) -> str:
    dr, dc = DIRS[direction]
    for _ in range(k % 4):
        dr, dc = -dc, dr
    for name, vec in DIRS.items():
        if vec == (dr, dc):
            return name
    raise AssertionError("unreachable direction rotation")


def _open_right_patch(
    rng: np.random.Generator,
    *,
    thickness: int,
    inner_h: int,
    inner_w: int,
    color: int,
    exit_offset: int,
) -> tuple[np.ndarray, ContainerMeta]:
    h = inner_h + 2 * thickness
    w = inner_w + thickness
    patch = np.full((h, w), BG, dtype=int)
    patch[:thickness, :] = GREY
    patch[-thickness:, :] = GREY
    patch[:, :thickness] = GREY

    seed_r = thickness + int(rng.integers(inner_h))
    seed_c = thickness + inner_w - 1 - exit_offset
    patch[seed_r, seed_c] = color
    return patch, ContainerMeta(seed_r, seed_c, color, "right", thickness)


def _rotate_patch_and_meta(
    patch: np.ndarray,
    metas: list[ContainerMeta],
    k: int,
) -> tuple[np.ndarray, list[ContainerMeta]]:
    k %= 4
    if k == 0:
        return patch.copy(), metas.copy()
    h, w = patch.shape
    rot = np.rot90(patch, k)
    new_metas: list[ContainerMeta] = []
    for meta in metas:
        rr, cc = _rotate_point(meta.seed_r, meta.seed_c, h, w, k)
        new_metas.append(
            ContainerMeta(rr, cc, meta.color, _rotate_dir(meta.direction, k), meta.thickness)
        )
    return rot, new_metas


def _compose_group(
    pieces: list[tuple[np.ndarray, list[ContainerMeta], int, int]],
) -> tuple[np.ndarray, list[ContainerMeta]]:
    min_r = min(r0 for _patch, _metas, r0, _c0 in pieces)
    min_c = min(c0 for _patch, _metas, _r0, c0 in pieces)
    max_r = max(r0 + patch.shape[0] for patch, _metas, r0, _c0 in pieces)
    max_c = max(c0 + patch.shape[1] for patch, _metas, _r0, c0 in pieces)

    h = max_r - min_r
    w = max_c - min_c
    group = np.full((h, w), BG, dtype=int)
    metas: list[ContainerMeta] = []

    for patch, patch_metas, r0, c0 in pieces:
        rr = r0 - min_r
        cc = c0 - min_c
        region = group[rr : rr + patch.shape[0], cc : cc + patch.shape[1]]
        overlap = (region != BG) & (patch != BG)
        assert not np.any(overlap), "group pieces must not overlap"
        region[patch != BG] = patch[patch != BG]
        for meta in patch_metas:
            metas.append(
                ContainerMeta(
                    rr + meta.seed_r,
                    cc + meta.seed_c,
                    meta.color,
                    meta.direction,
                    meta.thickness,
                )
            )
    return group, metas


def _group_output_box(
    group: np.ndarray,
    metas: list[ContainerMeta],
) -> tuple[int, int, int, int]:
    pad = 3
    h, w = group.shape
    canvas = np.full((h + 2 * pad, w + 2 * pad), BG, dtype=int)
    canvas[pad : pad + h, pad : pad + w] = group
    shifted = [
        ContainerMeta(
            pad + meta.seed_r,
            pad + meta.seed_c,
            meta.color,
            meta.direction,
            meta.thickness,
        )
        for meta in metas
    ]
    out = _apply_growth(canvas, shifted)
    coords = np.argwhere(out != BG)
    r0, c0, r1, c1 = _bbox(coords)
    return r0 - pad, c0 - pad, r1 - pad, c1 - pad


def _make_clear_group(
    rng: np.random.Generator,
    color: int,
) -> tuple[np.ndarray, list[ContainerMeta]]:
    thickness = int(rng.integers(1, 4))
    inner_h = int(rng.integers(2, 6))
    inner_w = int(rng.integers(2, 6))
    exit_offset = int(rng.integers(0, min(thickness, inner_w)))
    patch, meta = _open_right_patch(
        rng,
        thickness=thickness,
        inner_h=inner_h,
        inner_w=inner_w,
        color=color,
        exit_offset=exit_offset,
    )
    k = int(rng.integers(0, 4))
    return _rotate_patch_and_meta(patch, [meta], k)


def _make_blocked_group(
    rng: np.random.Generator,
    colors: tuple[int, int],
) -> tuple[np.ndarray, list[ContainerMeta]]:
    source_color, blocker_color = colors

    source_t = int(rng.integers(2, 4))
    source_inner_h = int(rng.integers(2, 6))
    source_inner_w = int(rng.integers(2, 6))
    stop_len = int(rng.integers(1, source_t))
    exit_offset = int(rng.integers(0, stop_len))

    src_patch, src_meta = _open_right_patch(
        rng,
        thickness=source_t,
        inner_h=source_inner_h,
        inner_w=source_inner_w,
        color=source_color,
        exit_offset=exit_offset,
    )

    blocker_t = int(rng.integers(1, 4))
    blocker_inner_h = int(rng.integers(2, 6))
    blocker_inner_w = int(rng.integers(2, 6))
    blocker_exit = int(rng.integers(0, min(blocker_t, blocker_inner_w)))
    blocker_dir = "up" if bool(rng.integers(0, 2)) else "down"

    blocker_base, blocker_meta_base = _open_right_patch(
        rng,
        thickness=blocker_t,
        inner_h=blocker_inner_h,
        inner_w=blocker_inner_w,
        color=blocker_color,
        exit_offset=blocker_exit,
    )
    blocker_patch, blocker_metas = _rotate_patch_and_meta(
        blocker_base,
        [blocker_meta_base],
        3 if blocker_dir == "down" else 1,
    )
    blocker_meta = blocker_metas[0]

    wall_col = src_meta.seed_c + stop_len + 1
    rel_row = int(rng.integers(0, blocker_patch.shape[0]))
    blocker_r0 = src_meta.seed_r - rel_row
    blocker_c0 = wall_col

    group, metas = _compose_group(
        [
            (src_patch, [src_meta], 0, 0),
            (blocker_patch, [blocker_meta], blocker_r0, blocker_c0),
        ]
    )
    k = int(rng.integers(0, 4))
    return _rotate_patch_and_meta(group, metas, k)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ac1 + gap < bc0 or bc1 + gap < ac0 or ar1 + gap < br0 or br1 + gap < ar0


def _place_groups(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    groups: list[tuple[np.ndarray, list[ContainerMeta]]],
) -> tuple[np.ndarray, list[ContainerMeta]] | None:
    h, w = grid_shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[np.ndarray, list[ContainerMeta], int, int]] = []

    for group, metas in groups:
        gh, gw = group.shape
        if gh > h - 2 * EDGE_MARGIN or gw > w - 2 * EDGE_MARGIN:
            return None
        fr0, fc0, fr1, fc1 = _group_output_box(group, metas)

        candidates: list[tuple[int, int]] = []
        for r0 in range(EDGE_MARGIN, h - EDGE_MARGIN - gh + 1):
            for c0 in range(EDGE_MARGIN, w - EDGE_MARGIN - gw + 1):
                box = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
                if box[0] < 0 or box[1] < 0 or box[2] >= h or box[3] >= w:
                    continue
                if all(_boxes_separated(box, other, OUTPUT_GAP) for other in placed_boxes):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        placed_boxes.append((r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1))
        placements.append((group, metas, r0, c0))

    grid = np.full(grid_shape, BG, dtype=int)
    all_metas: list[ContainerMeta] = []
    for group, metas, r0, c0 in placements:
        region = grid[r0 : r0 + group.shape[0], c0 : c0 + group.shape[1]]
        overlap = (region != BG) & (group != BG)
        assert not np.any(overlap), "placed groups must not overlap"
        region[group != BG] = group[group != BG]
        for meta in metas:
            all_metas.append(
                ContainerMeta(
                    r0 + meta.seed_r,
                    c0 + meta.seed_c,
                    meta.color,
                    meta.direction,
                    meta.thickness,
                )
            )
    return grid, all_metas


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    primary_blocked = bool(rng.integers(0, 2))
    extra_clear = int(rng.integers(0, EXTRA_CLEAR_MAX + 1))
    total_containers = 2 + extra_clear if primary_blocked else 1 + extra_clear
    colors = [int(c) for c in rng.choice(PALETTE, size=total_containers, replace=False)]

    groups: list[tuple[np.ndarray, list[ContainerMeta]]] = []
    color_index = 0
    if primary_blocked:
        groups.append(_make_blocked_group(rng, (colors[color_index], colors[color_index + 1])))
        color_index += 2
    else:
        groups.append(_make_clear_group(rng, colors[color_index]))
        color_index += 1

    for _ in range(extra_clear):
        groups.append(_make_clear_group(rng, colors[color_index]))
        color_index += 1

    order = list(rng.permutation(len(groups)))
    groups = [groups[i] for i in order]

    grid_h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid_w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    placed = _place_groups(rng, (grid_h, grid_w), groups)
    if placed is None:
        raise AssertionError("failed to place generated groups")
    inp, metas = placed
    expected = _apply_growth(inp, metas)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        try:
            inp, out = _generate_case(rng)
        except AssertionError:
            continue
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid input")
