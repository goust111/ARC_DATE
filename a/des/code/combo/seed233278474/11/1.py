from __future__ import annotations

import itertools

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
SIDE_ORDER = ("up", "right", "down", "left")


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    stack = [tuple(map(int, coords[0]))]
    seen = {stack[0]}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    return comps


def _corner_score(side: str, bbox: tuple[int, int, int, int], height: int, width: int) -> int:
    r0, c0, r1, c1 = bbox
    if side == "up":
        return int(r0 + c0)
    if side == "right":
        return int(r0 + (width - 1 - c1))
    if side == "down":
        return int((height - 1 - r1) + (width - 1 - c1))
    return int((height - 1 - r1) + c0)


def _extract_templates(grid: np.ndarray) -> dict[str, np.ndarray]:
    comps = _components_of_color(grid, TEMPLATE_COLOR)
    assert len(comps) >= 4, "expected four gray corner templates"
    height, width = grid.shape

    best_perm: tuple[int, ...] | None = None
    best_score: int | None = None
    for perm in itertools.permutations(range(len(comps)), 4):
        score = sum(_corner_score(side, comps[idx]["bbox"], height, width) for side, idx in zip(SIDE_ORDER, perm))
        if best_score is None or score < best_score:
            best_score = score
            best_perm = perm

    assert best_perm is not None
    return {side: comps[idx]["mask"] for side, idx in zip(SIDE_ORDER, best_perm)}


def _parse_shell(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if not np.all(sub[1:-1, 1:-1] == BG):
        return None
    if np.count_nonzero(sub == color) != int(border.sum()) - 1:
        return None

    missing = np.argwhere(border & (sub != color))
    if missing.shape[0] != 1:
        return None
    mr, mc = map(int, missing[0])
    if int(sub[mr, mc]) != BG:
        return None
    if (mr in (0, h - 1)) and (mc in (0, w - 1)):
        return None

    if mr == 0:
        side = "up"
    elif mr == h - 1:
        side = "down"
    elif mc == 0:
        side = "left"
    elif mc == w - 1:
        side = "right"
    else:
        return None
    return side, bbox


def _stamp_template(out: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    th, tw = mask.shape
    assert th <= inner_h and tw <= inner_w, "template must fit inside shell cavity"
    sr = r0 + 1 + (inner_h - th) // 2
    sc = c0 + 1 + (inner_w - tw) // 2
    out[sr : sr + th, sc : sc + tw][mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a corner template from the shell gap direction and stamp it into the shell interior.

    Input structure:
    - Black background.
    - Four gray corner templates: top-left=up, top-right=right, bottom-right=down, bottom-left=left.
    - Several one-pixel-thick hollow rectangular shells, each with exactly one missing border pixel.

    Rule:
    - Detect which side of each shell contains the single-pixel gap.
    - Use that side as a lookup key into the corner templates.
    - Center the chosen template inside the shell cavity and recolor it to the shell color.
    - Preserve the original corner templates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    templates = _extract_templates(g)
    out = g.copy()
    for color in map(int, np.unique(g)):
        if color in (BG, TEMPLATE_COLOR):
            continue
        for comp in _components_of_color(g, color):
            parsed = _parse_shell(g, color, comp["bbox"])
            if parsed is None:
                continue
            side, bbox = parsed
            _stamp_template(out, bbox, templates[side], color)
    return out


def _all_template_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for h in range(1, 4):
        for w in range(1, 4):
            for bits in range(1, 1 << (h * w)):
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(h * w):
                    if (bits >> idx) & 1:
                        mask[idx // w, idx % w] = True
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if not _connected(mask):
                    continue
                key = (mask.shape, mask.tobytes())
                if key not in seen:
                    seen.add(key)
                    masks.append(mask)
    return masks


TEMPLATE_LIBRARY = _all_template_masks()


def _sample_templates(rng: np.random.Generator) -> dict[str, np.ndarray]:
    for _ in range(200):
        idx = rng.choice(len(TEMPLATE_LIBRARY), size=4, replace=False)
        masks = [TEMPLATE_LIBRARY[int(i)] for i in idx]
        signatures = {(mask.shape, mask.tobytes()) for mask in masks}
        if len(signatures) < 4:
            continue
        if len({mask.shape for mask in masks}) < 2:
            continue
        return {side: mask.copy() for side, mask in zip(SIDE_ORDER, masks)}
    raise RuntimeError("failed to sample diverse corner templates")


def _expanded_slice(bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int) -> tuple[slice, slice]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        slice(max(0, r0 - pad), min(h, r1 + pad + 1)),
        slice(max(0, c0 - pad), min(w, c1 + pad + 1)),
    )


def _mark_bbox(occ: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    rs, cs = _expanded_slice(bbox, occ.shape, pad)
    occ[rs, cs] = True


def _bbox_free(occ: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    rs, cs = _expanded_slice(bbox, occ.shape, pad)
    return not bool(occ[rs, cs].any())


def _place_templates(grid: np.ndarray, templates: dict[str, np.ndarray], occ: np.ndarray) -> None:
    h, w = grid.shape
    placements = {
        "up": (0, 0),
        "right": (0, w - templates["right"].shape[1]),
        "down": (h - templates["down"].shape[0], w - templates["down"].shape[1]),
        "left": (h - templates["left"].shape[0], 0),
    }
    for side, (r0, c0) in placements.items():
        mask = templates[side]
        th, tw = mask.shape
        grid[r0 : r0 + th, c0 : c0 + tw][mask] = TEMPLATE_COLOR
        _mark_bbox(occ, (r0, c0, r0 + th - 1, c0 + tw - 1), pad=1)


def _draw_shell(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, side: str, rng: np.random.Generator) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if side == "up":
        grid[r0, int(rng.integers(c0 + 1, c1))] = BG
    elif side == "down":
        grid[r1, int(rng.integers(c0 + 1, c1))] = BG
    elif side == "left":
        grid[int(rng.integers(r0 + 1, r1)), c0] = BG
    else:
        grid[int(rng.integers(r0 + 1, r1)), c1] = BG


def _sample_directions(rng: np.random.Generator, count: int) -> list[str]:
    while True:
        directions = [SIDE_ORDER[int(i)] for i in rng.integers(0, 4, size=count)]
        if count == 1 or len(set(directions)) >= 2:
            return directions


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_shells = int(rng.integers(2, 7))
    height = int(rng.integers(max(20, 14 + n_shells), 31))
    width = int(rng.integers(max(20, 14 + n_shells), 31))
    grid = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    templates = _sample_templates(rng)
    _place_templates(grid, templates, occ)

    palette = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
    if n_shells > len(palette):
        return None
    rng.shuffle(palette)
    colors = [int(color) for color in palette[:n_shells]]
    directions = _sample_directions(rng, n_shells)

    shells: list[tuple[tuple[int, int, int, int], int, str]] = []
    for color, side in zip(colors, directions):
        placed = False
        min_inner_h, min_inner_w = templates[side].shape
        for _ in range(200):
            outer_h = int(rng.integers(max(5, min_inner_h + 2), 11))
            outer_w = int(rng.integers(max(5, min_inner_w + 2), 11))
            if outer_h >= height - 2 or outer_w >= width - 2:
                continue
            r0 = int(rng.integers(1, height - outer_h))
            c0 = int(rng.integers(1, width - outer_w))
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _bbox_free(occ, bbox, pad=1):
                continue
            _draw_shell(grid, bbox, color, side, rng)
            _mark_bbox(occ, bbox, pad=1)
            shells.append((bbox, color, side))
            placed = True
            break
        if not placed:
            return None

    if len({tuple(bbox) for bbox, _color, _side in shells}) != n_shells:
        return None
    if np.count_nonzero(grid == TEMPLATE_COLOR) == 0:
        return None

    expected = grid.copy()
    for bbox, color, side in shells:
        _stamp_template(expected, bbox, templates[side], color)

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.array_equal(out, expected):
        return None

    shell_count = 0
    for color in colors:
        for comp in _components_of_color(grid, color):
            if _parse_shell(grid, color, comp["bbox"]) is not None:
                shell_count += 1
    if shell_count != n_shells:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
