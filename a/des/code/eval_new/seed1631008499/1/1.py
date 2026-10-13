from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500
COMPONENT_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in COMPONENT_NEIGHBORS:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _is_frame_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(coords) == expected


def _point_in_frame_interior(frame_bbox: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = frame_bbox
    r, c = point
    return r0 < r < r1 and c0 < c < c1


def _rotate_coords(patch: np.ndarray, value: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(patch == value)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed seed layout to a grey-marked stamp, rotate it, recolor the markers, and stamp it into the frame.

    Input structure:
    - Black background fills the grid.
    - Several single-color rectangular frames appear as 1-pixel borders; each frame interior is mostly black and contains 1-3 isolated colored seed pixels.
    - Separate compact stamp templates appear elsewhere on the background; each template is a connected multicolor pattern containing some grey placeholder pixels.

    Rule:
    - For each frame, consider templates whose number of grey placeholders equals the number of seeds in that frame.
    - Rotate the matching template by 0/90/180/270 degrees so that its grey placeholder coordinates exactly match the seed coordinates when both are measured from the frame interior's top-left corner.
    - Replace those grey placeholders with the corresponding seed colors, erase the original standalone template, and stamp the rotated colored template centered inside the frame while keeping the border.

    Key insight:
    - Seed count chooses the candidate template, but the seed coordinate pattern fixes the unique rotation; grey cells are only placeholders and inherit the frame's seed colors before stamping.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    comps = _components(g != BG)
    frames = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    template_components: list[list[tuple[int, int]]] = []

    for comp in comps:
        if _is_frame_component(g, comp):
            bbox = _bbox(comp)
            frames.append(
                {
                    "bbox": bbox,
                    "color": int(g[comp[0][0], comp[0][1]]),
                }
            )
            frame_bboxes.append(bbox)
        else:
            template_components.append(comp)

    templates = []
    for comp in template_components:
        if any(_point_in_frame_interior(bbox, cell) for bbox in frame_bboxes for cell in comp):
            continue
        patch, bbox = _crop_component(g, comp)
        grey_count = int(np.sum(patch == GREY))
        if grey_count == 0:
            continue
        templates.append(
            {
                "patch": patch,
                "bbox": bbox,
                "grey_count": grey_count,
            }
        )

    for template in templates:
        r0, c0, r1, c1 = template["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    used_templates: set[int] = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = [
            (int(sr), int(sc), int(interior[sr, sc]))
            for sr, sc in np.argwhere(interior != BG)
        ]
        seeds.sort()
        seed_positions = {(sr, sc) for sr, sc, _ in seeds}
        matches = []
        for idx, template in enumerate(templates):
            if idx in used_templates or template["grey_count"] != len(seeds):
                continue
            for rot in range(4):
                rotated = np.rot90(template["patch"], rot)
                grey_positions = set(_rotate_coords(rotated, GREY))
                if grey_positions == seed_positions:
                    matches.append((idx, rotated))
        assert len(matches) == 1, "each frame must match exactly one template orientation"

        template_idx, rotated = matches[0]
        used_templates.add(template_idx)

        filled = rotated.copy()
        for sr, sc, color in seeds:
            filled[sr, sc] = color

        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        ih, iw = interior.shape
        ph, pw = filled.shape
        top = r0 + 1 + (ih - ph) // 2
        left = c0 + 1 + (iw - pw) // 2
        sub = out[top : top + ph, left : left + pw]
        mask = filled != BG
        sub[mask] = filled[mask]

    return out


def _random_connected_cells(rng: random.Random, h: int, w: int, target: int) -> set[tuple[int, int]]:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = set()
    frontier.update((nr, nc) for nr, nc in _neighbors4(*start) if 0 <= nr < h and 0 <= nc < w)
    while frontier and len(cells) < target:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _rotation_signatures(patch: np.ndarray) -> set[tuple[tuple[int, int], frozenset[tuple[int, int]]]]:
    sigs = set()
    for rot in range(4):
        rotated = np.rot90(patch, rot)
        sigs.add((rotated.shape, frozenset(_rotate_coords(rotated, GREY))))
    return sigs


def _random_template(rng: random.Random, grey_count: int) -> np.ndarray | None:
    for _ in range(200):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        max_cells = min(h * w, grey_count + 8)
        target = rng.randint(grey_count + 2, max_cells)
        cells = _random_connected_cells(rng, h, w, target)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        if len(coords) < grey_count + 2:
            continue

        for _ in range(80):
            grey_cells = set(rng.sample(coords, grey_count))
            patch = np.zeros(mask.shape, dtype=int)
            body_cells = [coord for coord in coords if coord not in grey_cells]
            if not body_cells:
                continue
            color_count = rng.randint(1, min(3, len(PALETTE)))
            body_colors = rng.sample(PALETTE, color_count)
            for r, c in body_cells:
                patch[r, c] = rng.choice(body_colors)
            for r, c in grey_cells:
                patch[r, c] = GREY
            if len(_rotation_signatures(patch)) != 4:
                continue
            return patch
    return None


def _build_frame_pair(
    rng: random.Random,
    template_patch: np.ndarray,
    rotation: int,
    frame_color: int,
    seed_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    rotated = np.rot90(template_patch, rotation)
    ph, pw = rotated.shape
    ih = ph + rng.randint(2, 6)
    iw = pw + rng.randint(2, 6)

    input_patch = np.zeros((ih + 2, iw + 2), dtype=int)
    input_patch[0, :] = frame_color
    input_patch[-1, :] = frame_color
    input_patch[:, 0] = frame_color
    input_patch[:, -1] = frame_color

    grey_coords = _rotate_coords(rotated, GREY)
    rng.shuffle(grey_coords)
    color_map = {coord: color for coord, color in zip(grey_coords, seed_colors)}
    for (r, c), color in color_map.items():
        input_patch[1 + r, 1 + c] = color

    output_patch = np.zeros_like(input_patch)
    output_patch[0, :] = frame_color
    output_patch[-1, :] = frame_color
    output_patch[:, 0] = frame_color
    output_patch[:, -1] = frame_color

    filled = rotated.copy()
    for (r, c), color in color_map.items():
        filled[r, c] = color
    top = 1 + (ih - ph) // 2
    left = 1 + (iw - pw) // 2
    sub = output_patch[top : top + ph, left : left + pw]
    mask = filled != BG
    sub[mask] = filled[mask]
    return input_patch, output_patch


def _place_patches(
    rng: random.Random,
    canvas_shape: tuple[int, int],
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        ph, pw = patches[idx].shape
        if ph + 2 > h or pw + 2 > w:
            return None
        positions = [
            (r, c)
            for r in range(1, h - ph)
            for c in range(1, w - pw)
        ]
        rng.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            rlo = r0 - 1
            rhi = r0 + ph + 1
            clo = c0 - 1
            chi = c0 + pw + 1
            if occupied[rlo:rhi, clo:chi].any():
                continue
            occupied[rlo:rhi, clo:chi] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [placement for placement in placements if placement is not None]


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int):
    sub = dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BG
    sub[mask] = patch[mask]


def _random_crop_bounds(rng: random.Random, mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    top_extra = rng.randint(1, min(4, r0) if r0 > 0 else 1)
    left_extra = rng.randint(1, min(4, c0) if c0 > 0 else 1)
    bottom_space = mask.shape[0] - 1 - r1
    right_space = mask.shape[1] - 1 - c1
    bottom_extra = rng.randint(1, min(4, bottom_space) if bottom_space > 0 else 1)
    right_extra = rng.randint(1, min(4, right_space) if right_space > 0 else 1)
    return r0 - top_extra, r1 + bottom_extra + 1, c0 - left_extra, c1 + right_extra + 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([1, 2, 3])
    counts = rng.sample([1, 2, 3], pair_count)
    rng.shuffle(counts)

    frame_colors = rng.sample(PALETTE, pair_count)
    pairs = []
    for idx, grey_count in enumerate(counts):
        template_patch = _random_template(rng, grey_count)
        if template_patch is None:
            return None
        template_colors = {int(v) for v in np.unique(template_patch)} - {BG, GREY}
        available_seed_colors = [c for c in PALETTE if c not in template_colors and c != frame_colors[idx]]
        if len(available_seed_colors) < grey_count:
            return None
        seed_colors = rng.sample(available_seed_colors, grey_count)
        rotation = rng.randrange(4)
        frame_input, frame_output = _build_frame_pair(rng, template_patch, rotation, frame_colors[idx], seed_colors)
        pairs.append(
            {
                "template": template_patch,
                "frame_input": frame_input,
                "frame_output": frame_output,
            }
        )

    input_patches = [pair["frame_input"] for pair in pairs] + [pair["template"] for pair in pairs]
    placements = _place_patches(rng, (30, 30), input_patches)
    if placements is None:
        return None

    input_canvas = np.zeros((30, 30), dtype=int)
    output_canvas = np.zeros((30, 30), dtype=int)
    for idx, pair in enumerate(pairs):
        fr, fc = placements[idx]
        tr, tc = placements[pair_count + idx]
        _paste(input_canvas, pair["frame_input"], fr, fc)
        _paste(output_canvas, pair["frame_output"], fr, fc)
        _paste(input_canvas, pair["template"], tr, tc)

    non_bg = input_canvas != BG
    if not np.any(non_bg):
        return None
    r0, r1, c0, c1 = _random_crop_bounds(rng, non_bg)
    input_grid = input_canvas[r0:r1, c0:c1].copy()
    expected = output_canvas[r0:r1, c0:c1].copy()

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(input_grid, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None and grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
