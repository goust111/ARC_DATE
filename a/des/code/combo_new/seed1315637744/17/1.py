from __future__ import annotations

from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = DIR4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

# --------------- generator config ---------------

GEN_RETRIES = 400
PAIR_COUNT_RANGE = (2, 5)  # 2..4
GRID_RANGES = {
    2: ((18, 25), (18, 25)),
    3: ((21, 28), (21, 28)),
    4: ((24, 30), (24, 30)),
}
SHAPE_H_RANGE = (2, 6)
SHAPE_W_RANGE = (2, 7)
SHAPE_AREA_RATIO_RANGE = (0.45, 0.8)
FRAME_EXTRA_RANGE = (0, 4)
PLACEMENT_GAP = 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray, *, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if int(connectivity) == 8 else DIR4
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _c in coords]
    cs = [int(c) for _r, c in coords]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _touches_grid_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(int(r) in (0, h - 1) or int(c) in (0, w - 1) for r, c in coords)


def _is_rectangular_frame(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return len(coords) == int(border.sum()) and bool(np.array_equal(patch == int(color), border))


def _detect_frames(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    frames: dict[int, tuple[int, int, int, int]] = {}
    for color in map(int, np.unique(grid)):
        if int(color) == int(bg):
            continue
        for comp in _components(grid == int(color), connectivity=4):
            if not _is_rectangular_frame(grid, comp, int(color)):
                continue
            assert int(color) not in frames, "frame colors must be unique"
            frames[int(color)] = _bbox(comp)
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey loop's trapped marker color to the same-color frame and copy the loop interior there.

    Input structure:
    - A dominant background color fills most of the grid.
    - Several grey 1-pixel closed loops enclose irregular background cavities; each cavity contains exactly one colored marker pixel.
    - Elsewhere, hollow rectangular frames use distinct colors that match the marker colors one-to-one.

    Rule:
    - Keep every grey loop outline and every colored rectangular frame border unchanged.
    - For each grey-enclosed cavity, read its unique colored marker color C.
    - Find the hollow rectangular frame whose border color is also C.
    - Take the full cavity shape inside the grey loop (all non-grey cells in that enclosed component, including background cells and the marker position), paint that shape with color C, and paste it into the frame interior with top-left alignment.
    - Remove the marker from the grey loop interior by restoring it to the background color.

    Key insight:
    - The single colored pixel inside each loop is an address label: it does not stay in place, but tells which same-color frame should receive the loop's enclosed interior mask.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(_most_common_color(g))
    out = g.copy()

    frames = _detect_frames(g, int(bg))
    assert frames, "expected at least one colored frame"

    non_bg_colors = set(map(int, np.unique(g))) - {int(bg)}
    wall_candidates = list(non_bg_colors - set(frames))
    assert len(wall_candidates) == 1, "expected one shared loop-outline color"
    wall = int(wall_candidates[0])

    for comp in _components(g != int(wall), connectivity=4):
        if _touches_grid_border(comp, h, w):
            continue

        markers = [(int(r), int(c)) for r, c in comp if int(g[r, c]) != int(bg)]
        if len(markers) != 1:
            continue
        mr, mc = markers[0]
        color = int(g[mr, mc])
        if int(color) not in frames:
            continue

        r0, c0, r1, c1 = _bbox(comp)
        shape = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        for rr, cc in comp:
            shape[int(rr - r0), int(cc - c0)] = True

        out[mr, mc] = int(bg)

        fr0, fc0, fr1, fc1 = frames[int(color)]
        inner_h = int(fr1 - fr0 - 1)
        inner_w = int(fc1 - fc0 - 1)
        sh, sw = map(int, shape.shape)
        assert sh <= inner_h and sw <= inner_w, "frame interior must fit the copied cavity"

        target = out[fr0 + 1 : fr0 + 1 + sh, fc0 + 1 : fc0 + 1 + sw]
        target[shape] = int(color)

    return out


def _mask_has_hole(mask: np.ndarray) -> bool:
    inv = ~mask
    for comp in _components(inv, connectivity=4):
        if not _touches_grid_border(comp, int(mask.shape[0]), int(mask.shape[1])):
            return True
    return False


def _random_shape_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(*SHAPE_H_RANGE))
        w = int(rng.integers(*SHAPE_W_RANGE))
        total = int(h * w)
        min_area = max(4, int(np.ceil(total * SHAPE_AREA_RATIO_RANGE[0])))
        max_area = min(total - 1, int(np.floor(total * SHAPE_AREA_RATIO_RANGE[1])))
        if max_area < min_area:
            continue
        area = int(rng.integers(min_area, max_area + 1))

        seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {seed}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])

        rs = [int(r) for r, _c in cells]
        cs = [int(c) for _r, c in cells]
        r0, c0 = int(min(rs)), int(min(cs))
        mask = np.zeros((int(max(rs) - r0 + 1), int(max(cs) - c0 + 1)), dtype=bool)
        for r, c in cells:
            mask[int(r - r0), int(c - c0)] = True

        if min(mask.shape) < 2:
            continue
        if _mask_has_hole(mask):
            continue
        if bool(mask.all()):
            continue
        if len(_components(mask, connectivity=4)) != 1:
            continue
        return mask
    raise RuntimeError("failed to build a loop interior mask")


def _make_loop_patch(
    mask: np.ndarray, bg: int, wall: int, color: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = map(int, mask.shape)
    patch = np.full((h + 2, w + 2), int(bg), dtype=int)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    marker_r, marker_c = cells[int(rng.integers(0, len(cells)))]

    for r, c in cells:
        rr = int(r + 1)
        cc = int(c + 1)
        if (int(r), int(c)) == (int(marker_r), int(marker_c)):
            patch[rr, cc] = int(color)
        for dr, dc in DIR4:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < h + 2 and 0 <= nc < w + 2:
                inside_mask = 1 <= nr <= h and 1 <= nc <= w and bool(mask[nr - 1, nc - 1])
                if not inside_mask:
                    patch[nr, nc] = int(wall)

    wall_comps = _components(patch == int(wall), connectivity=8)
    assert len(wall_comps) == 1, "loop outline must stay connected"
    return patch, (int(marker_r + 1), int(marker_c + 1))


def _make_frame_patch(inner_h: int, inner_w: int, bg: int, color: int) -> np.ndarray:
    patch = np.full((int(inner_h + 2), int(inner_w + 2)), int(bg), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _boxes_overlap(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + int(gap) < br0
        or br1 + int(gap) < ar0
        or ac1 + int(gap) < bc0
        or bc1 + int(gap) < ac0
    )


def _place_objects(
    rng: np.random.Generator, grid_h: int, grid_w: int, patches: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        ph, pw = map(int, patches[idx].shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(int(grid_h - ph + 1)):
            for c0 in range(int(grid_w - pw + 1)):
                box = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
                if any(_boxes_overlap(box, other, PLACEMENT_GAP) for other in placed_boxes):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        pos = candidates[int(rng.integers(0, len(candidates)))]
        positions[idx] = pos
        placed_boxes.append((int(pos[0]), int(pos[1]), int(pos[0] + ph - 1), int(pos[1] + pw - 1)))

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    ph, pw = map(int, patch.shape)
    sub = grid[r0 : r0 + ph, c0 : c0 + pw]
    if transparent is None:
        sub[:, :] = patch
    else:
        mask = patch != int(transparent)
        sub[mask] = patch[mask]


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_pairs = int(rng.integers(*PAIR_COUNT_RANGE))
    (h_range, w_range) = GRID_RANGES[int(n_pairs)]

    palette = list(map(int, rng.choice(np.arange(10), size=int(n_pairs + 2), replace=False)))
    bg = int(palette[0])
    wall = int(palette[1])
    pair_colors = [int(c) for c in palette[2:]]

    loop_patches: list[np.ndarray] = []
    frame_patches: list[np.ndarray] = []
    loop_masks: list[np.ndarray] = []
    loop_markers: list[tuple[int, int]] = []

    for color in pair_colors:
        mask = _random_shape_mask(rng)
        loop_patch, marker = _make_loop_patch(mask, int(bg), int(wall), int(color), rng)
        extra_h = int(rng.integers(*FRAME_EXTRA_RANGE))
        extra_w = int(rng.integers(*FRAME_EXTRA_RANGE))
        frame_inner_h = int(mask.shape[0] + extra_h)
        frame_inner_w = int(mask.shape[1] + extra_w)
        frame_patch = _make_frame_patch(frame_inner_h, frame_inner_w, int(bg), int(color))

        loop_masks.append(mask)
        loop_markers.append(marker)
        loop_patches.append(loop_patch)
        frame_patches.append(frame_patch)

    patches = loop_patches + frame_patches
    min_h = max(int(h_range[0]), max(int(p.shape[0]) for p in patches) + 2)
    min_w = max(int(w_range[0]), max(int(p.shape[1]) for p in patches) + 2)
    if min_h > int(h_range[1]) or min_w > int(w_range[1]):
        return None

    grid_h = int(rng.integers(min_h, int(h_range[1]) + 1))
    grid_w = int(rng.integers(min_w, int(w_range[1]) + 1))
    positions = _place_objects(rng, int(grid_h), int(grid_w), patches)
    if positions is None:
        return None

    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        _paint_patch(inp, patch, int(r0), int(c0))

    out = inp.copy()
    for idx, color in enumerate(pair_colors):
        loop_r0, loop_c0 = positions[idx]
        frame_r0, frame_c0 = positions[int(idx + n_pairs)]
        mr, mc = loop_markers[idx]
        out[int(loop_r0 + mr), int(loop_c0 + mc)] = int(bg)

        mask = loop_masks[idx]
        sh, sw = map(int, mask.shape)
        target = out[frame_r0 + 1 : frame_r0 + 1 + sh, frame_c0 + 1 : frame_c0 + 1 + sw]
        target[mask] = int(color)

    if _most_common_color(inp) != int(bg):
        return None
    if not bool(np.array_equal(main(inp), out)):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case(np.random.default_rng())
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
