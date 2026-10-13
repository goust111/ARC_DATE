from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class _Container:
    color: int
    bbox: tuple[int, int, int, int]
    border_coords: tuple[tuple[int, int], ...]

    @property
    def interior_shape(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0 - 1, c1 - c0 - 1)

    @property
    def interior_origin(self) -> tuple[int, int]:
        r0, c0, _r1, _c1 = self.bbox
        return r0 + 1, c0 + 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_frame_from_component(color: int, coords: np.ndarray) -> _Container | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    rel = coords - np.array([r0, c0], dtype=int)
    border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    if not bool(np.all(border)):
        return None

    expected_area = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != expected_area:
        return None

    return _Container(
        color=int(color),
        bbox=(r0, c0, r1, c1),
        border_coords=tuple((int(r), int(c)) for r, c in coords.tolist()),
    )


def _find_containers(g: np.ndarray) -> list[_Container]:
    candidates: list[_Container] = []
    for color in sorted(set(map(int, np.unique(g))) - {BG}):
        for coords in _components(g == color):
            container = _rect_frame_from_component(color, coords)
            if container is not None:
                candidates.append(container)

    top_level: list[_Container] = []
    for i, outer in enumerate(candidates):
        r0, c0, r1, c1 = outer.bbox
        nested = False
        for j, inner in enumerate(candidates):
            if i == j:
                continue
            ir0, ic0, ir1, ic1 = inner.bbox
            if r0 < ir0 and c0 < ic0 and ir1 < r1 and ic1 < c1:
                nested = True
                break
        if not nested:
            top_level.append(outer)

    assert top_level, "expected at least one top-level container frame"
    return top_level


def _interior_non_bg(g: np.ndarray, container: _Container) -> np.ndarray:
    r, c = container.interior_origin
    h, w = container.interior_shape
    return g[r : r + h, c : c + w] != BG


def _pipe_components(g: np.ndarray, containers: list[_Container]) -> list[tuple[int, np.ndarray]]:
    outside = np.ones(g.shape, dtype=bool)
    for container in containers:
        r0, c0, r1, c1 = container.bbox
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    pipes: list[tuple[int, np.ndarray]] = []
    for color in sorted(set(map(int, np.unique(g))) - {BG}):
        mask = (g == color) & outside
        for coords in _components(mask):
            pipes.append((int(color), coords))
    assert pipes, "expected at least one pipe component"
    return pipes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer each source container's interior pattern through its pipe into the paired empty container.

    Input structure:
    - The background is black.
    - Several one-pixel-thick monochromatic rectangular frames act as containers.
    - Some containers hold a non-black interior pattern, some are empty, and each connected pipe is a
      single-color orthogonal line outside the containers that touches exactly two container walls.

    Rule:
    - For every pipe, identify the touched container pair and determine which one is the source by checking
      which interior is non-empty.
    - Clear the source interior, then copy its non-black shape into the empty target interior with the same
      top-left offset relative to the inner wall.
    - Recolor the copied shape with the pipe color and crop any part that falls outside a smaller target.

    Key insight:
    - The pipe simultaneously defines container pairing and the recolor used after transport; the geometry of
      the pattern stays fixed relative to the container interior instead of being scaled or moved by centroid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    containers = _find_containers(g)
    frame_at = np.full(g.shape, -1, dtype=int)
    for idx, container in enumerate(containers):
        for r, c in container.border_coords:
            frame_at[r, c] = idx

    source_masks = [_interior_non_bg(g, container) for container in containers]
    out = g.copy()

    for pipe_color, coords in _pipe_components(g, containers):
        touched: set[int] = set()
        for r, c in coords.tolist():
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and frame_at[nr, nc] >= 0:
                    touched.add(int(frame_at[nr, nc]))

        assert len(touched) == 2, "each pipe must connect exactly two containers"
        a, b = sorted(touched)
        a_has_pattern = bool(source_masks[a].any())
        b_has_pattern = bool(source_masks[b].any())
        assert a_has_pattern != b_has_pattern, "each pipe must link one source and one empty target"
        src_id, dst_id = (a, b) if a_has_pattern else (b, a)

        src = containers[src_id]
        dst = containers[dst_id]
        src_mask = source_masks[src_id]
        dst_r0, dst_c0 = dst.interior_origin
        dst_h, dst_w = dst.interior_shape

        sr0, sc0 = src.interior_origin
        out[sr0 : sr0 + src_mask.shape[0], sc0 : sc0 + src_mask.shape[1]][src_mask] = BG

        for rr, cc in np.argwhere(src_mask):
            rr = int(rr)
            cc = int(cc)
            if rr < dst_h and cc < dst_w:
                out[dst_r0 + rr, dst_c0 + cc] = pipe_color

    return out


def _random_connected_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    area = h * w
    min_cells = 1 if area == 1 else 2
    target = rng.randint(min_cells, area)
    for _ in range(120):
        cells = {(rng.randrange(h), rng.randrange(w))}
        frontier = list(cells)
        while len(cells) < target:
            base_r, base_c = frontier[rng.randrange(len(frontier))]
            choices = [
                (nr, nc)
                for nr, nc in _neighbors4(base_r, base_c)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
            ]
            if choices:
                nxt = choices[rng.randrange(len(choices))]
                cells.add(nxt)
                frontier.append(nxt)
            else:
                frontier.pop(rng.randrange(len(frontier)))
                if not frontier:
                    frontier = list(cells)

        coords = np.asarray(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 != h or c1 - c0 + 1 != w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        return mask

    mask = np.ones((h, w), dtype=bool)
    return mask


def _sample_shape_and_target(
    src_h: int,
    src_w: int,
    crop_case: bool,
    size_cap: int,
    rng: random.Random,
) -> tuple[np.ndarray, int, int, int, int]:
    for _ in range(300):
        mask_h = rng.randint(1, src_h)
        mask_w = rng.randint(1, src_w)
        off_r = rng.randint(0, src_h - mask_h)
        off_c = rng.randint(0, src_w - mask_w)
        if crop_case:
            valid_targets = [
                (th, tw)
                for th in range(2, size_cap + 1)
                for tw in range(2, size_cap + 1)
                if off_r + mask_h > th or off_c + mask_w > tw
            ]
        else:
            valid_targets = [
                (th, tw)
                for th in range(2, size_cap + 1)
                for tw in range(2, size_cap + 1)
                if off_r + mask_h <= th and off_c + mask_w <= tw
            ]
        if not valid_targets:
            continue

        tgt_h, tgt_w = valid_targets[rng.randrange(len(valid_targets))]
        mask = _random_connected_mask(mask_h, mask_w, rng)
        return mask, off_r, off_c, tgt_h, tgt_w
    raise RuntimeError("failed to sample source pattern and target size")


def _frame_patch(frame_color: int, interior_h: int, interior_w: int) -> np.ndarray:
    patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    return patch


def _build_pair_patch(interior_cap: int, patch_limit: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    colors = list(range(1, 10))
    rng.shuffle(colors)

    same_frame_color = rng.choice([False, True])
    if same_frame_color:
        frame_a = colors.pop()
        frame_b = frame_a
    else:
        frame_a = colors.pop()
        frame_b = colors.pop()
    pattern_color = colors.pop()
    pipe_color = colors.pop()

    src_h = rng.randint(2, interior_cap)
    src_w = rng.randint(2, interior_cap)
    crop_case = rng.choice([False, True])
    shape_mask, off_r, off_c, tgt_h, tgt_w = _sample_shape_and_target(src_h, src_w, crop_case, interior_cap, rng)

    source = _frame_patch(frame_a, src_h, src_w)
    source[1 + off_r : 1 + off_r + shape_mask.shape[0], 1 + off_c : 1 + off_c + shape_mask.shape[1]][shape_mask] = (
        pattern_color
    )
    target = _frame_patch(frame_b, tgt_h, tgt_w)

    source_first = rng.choice([False, True])
    first = source if source_first else target
    second = target if source_first else source
    first_is_source = source_first

    layout = rng.choice(["h_straight", "h_bent", "v_straight", "v_bent"])
    margin = 1

    for _ in range(300):
        fh, fw = first.shape
        sh, sw = second.shape

        if layout == "h_straight":
            gap = rng.randint(2, 3)
            fr = rng.randint(1, fh - 2)
            sr = rng.randint(1, sh - 2)
            pipe_row = margin + max(fr, sr)
            first_r = pipe_row - fr
            second_r = pipe_row - sr
            first_c = margin
            second_c = first_c + fw + gap
            height = max(first_r + fh, second_r + sh) + margin
            width = second_c + sw + margin
            if height > patch_limit or width > patch_limit:
                continue
            patch = np.full((height, width), BG, dtype=int)
            patch[first_r : first_r + fh, first_c : first_c + fw] = first
            patch[second_r : second_r + sh, second_c : second_c + sw] = second
            patch[pipe_row, first_c + fw : second_c] = pipe_color

        elif layout == "h_bent":
            gap = rng.randint(2, 3)
            delta = rng.randint(1, 3)
            if rng.choice([False, True]):
                first_r = margin
                second_r = margin + delta
            else:
                second_r = margin
                first_r = margin + delta
            fr = rng.randint(1, fh - 2)
            sr = rng.randint(1, sh - 2)
            first_c = margin
            second_c = first_c + fw + gap
            height = max(first_r + fh, second_r + sh) + margin
            width = second_c + sw + margin
            if height > patch_limit or width > patch_limit:
                continue
            first_er = first_r + fr
            second_er = second_r + sr
            if first_er == second_er:
                continue
            bend_c = first_c + fw + rng.randint(1, gap - 1)
            patch = np.full((height, width), BG, dtype=int)
            patch[first_r : first_r + fh, first_c : first_c + fw] = first
            patch[second_r : second_r + sh, second_c : second_c + sw] = second
            patch[first_er, first_c + fw : bend_c + 1] = pipe_color
            lo, hi = sorted((first_er, second_er))
            patch[lo : hi + 1, bend_c] = pipe_color
            patch[second_er, bend_c : second_c] = pipe_color

        elif layout == "v_straight":
            gap = rng.randint(2, 3)
            fc = rng.randint(1, fw - 2)
            sc = rng.randint(1, sw - 2)
            pipe_col = margin + max(fc, sc)
            first_c = pipe_col - fc
            second_c = pipe_col - sc
            first_r = margin
            second_r = first_r + fh + gap
            height = second_r + sh + margin
            width = max(first_c + fw, second_c + sw) + margin
            if height > patch_limit or width > patch_limit:
                continue
            patch = np.full((height, width), BG, dtype=int)
            patch[first_r : first_r + fh, first_c : first_c + fw] = first
            patch[second_r : second_r + sh, second_c : second_c + sw] = second
            patch[first_r + fh : second_r, pipe_col] = pipe_color

        else:
            gap = rng.randint(2, 3)
            delta = rng.randint(1, 3)
            if rng.choice([False, True]):
                first_c = margin
                second_c = margin + delta
            else:
                second_c = margin
                first_c = margin + delta
            fc = rng.randint(1, fw - 2)
            sc = rng.randint(1, sw - 2)
            first_r = margin
            second_r = first_r + fh + gap
            height = second_r + sh + margin
            width = max(first_c + fw, second_c + sw) + margin
            if height > patch_limit or width > patch_limit:
                continue
            first_ec = first_c + fc
            second_ec = second_c + sc
            if first_ec == second_ec:
                continue
            bend_r = first_r + fh + rng.randint(1, gap - 1)
            patch = np.full((height, width), BG, dtype=int)
            patch[first_r : first_r + fh, first_c : first_c + fw] = first
            patch[second_r : second_r + sh, second_c : second_c + sw] = second
            patch[first_r + fh : bend_r + 1, first_ec] = pipe_color
            lo, hi = sorted((first_ec, second_ec))
            patch[bend_r, lo : hi + 1] = pipe_color
            patch[bend_r : second_r, second_ec] = pipe_color

        occupied = np.argwhere(patch != BG)
        r0, c0, r1, c1 = _bbox(occupied)
        patch = patch[r0 : r1 + 1, c0 : c1 + 1]

        if patch.shape[0] > patch_limit or patch.shape[1] > patch_limit:
            continue

        expected = main(patch)
        return patch, expected

    raise RuntimeError("failed to build pair patch")


def _composition_layout(pair_count: int, rng: random.Random) -> tuple[int, int, list[tuple[int, int]]]:
    if pair_count == 1:
        return 1, 1, [(0, 0)]
    if pair_count == 2:
        if rng.choice([False, True]):
            return 1, 2, [(0, 0), (0, 1)]
        return 2, 1, [(0, 0), (1, 0)]
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng.shuffle(cells)
    return 2, 2, cells[:3]


def _assemble_grid(pair_patches: list[tuple[np.ndarray, np.ndarray]], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    pair_count = len(pair_patches)
    rows, cols, slots = _composition_layout(pair_count, rng)
    max_patch_h = max(patch_in.shape[0] for patch_in, _patch_out in pair_patches)
    max_patch_w = max(patch_in.shape[1] for patch_in, _patch_out in pair_patches)
    gap = 1 if rows * cols > 1 else 0
    max_cell_h = (30 - (rows - 1) * gap) // rows
    max_cell_w = (30 - (cols - 1) * gap) // cols
    if max_patch_h > max_cell_h or max_patch_w > max_cell_w:
        raise RuntimeError("patches do not fit the chosen layout")

    slack_h = max(0, min(3, max_cell_h - max_patch_h))
    slack_w = max(0, min(3, max_cell_w - max_patch_w))
    cell_h = max_patch_h + rng.randint(0, slack_h)
    cell_w = max_patch_w + rng.randint(0, slack_w)
    total_h = rows * cell_h + (rows - 1) * gap
    total_w = cols * cell_w + (cols - 1) * gap
    assert total_h <= 30 and total_w <= 30

    inp = np.full((total_h, total_w), BG, dtype=int)
    out = np.full((total_h, total_w), BG, dtype=int)

    for (patch_in, patch_out), (cell_r, cell_c) in zip(pair_patches, slots, strict=False):
        ph, pw = patch_in.shape
        assert ph <= cell_h and pw <= cell_w
        r0 = cell_r * (cell_h + gap) + rng.randint(0, cell_h - ph)
        c0 = cell_c * (cell_w + gap) + rng.randint(0, cell_w - pw)
        inp[r0 : r0 + ph, c0 : c0 + pw] = patch_in
        out[r0 : r0 + ph, c0 : c0 + pw] = patch_out

    occ = np.argwhere(inp != BG)
    r0, c0, r1, c1 = _bbox(occ)
    top = rng.randint(0, min(2, r0))
    left = rng.randint(0, min(2, c0))
    bottom = rng.randint(0, min(2, total_h - 1 - r1))
    right = rng.randint(0, min(2, total_w - 1 - c1))
    r0 -= top
    c0 -= left
    r1 += bottom
    c1 += right
    return inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(1, 3)
    interior_cap = {1: 5, 2: 4, 3: 3}[pair_count]
    patch_limit = {1: 20, 2: 14, 3: 14}[pair_count]
    pair_patches: list[tuple[np.ndarray, np.ndarray]] = []

    try:
        for _ in range(pair_count):
            patch_in, patch_out = _build_pair_patch(interior_cap, patch_limit, rng)
            pair_patches.append((patch_in, patch_out))
    except RuntimeError:
        return None

    try:
        inp, out = _assemble_grid(pair_patches, rng)
    except (AssertionError, RuntimeError):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid transport-containers input")
