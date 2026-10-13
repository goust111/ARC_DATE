from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                colors.add(int(grid[r, c]))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": bbox,
                    "area": len(coords),
                    "colors": colors,
                }
            )
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    region = dst[top : top + h, left : left + w]
    mask = patch != bg
    region[mask] = patch[mask]


def _is_contiguous(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen[start] = True
    reached = 0
    while stack:
        r, c = stack.pop()
        reached += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return reached == int(mask.sum())


def _diag_map(legend: np.ndarray) -> dict[int, int]:
    return {
        int(legend[0, 0]): int(legend[1, 1]),
        int(legend[1, 1]): int(legend[0, 0]),
        int(legend[0, 1]): int(legend[1, 0]),
        int(legend[1, 0]): int(legend[0, 1]),
    }


def _row_map(legend: np.ndarray) -> dict[int, int]:
    return {
        int(legend[0, 0]): int(legend[0, 1]),
        int(legend[0, 1]): int(legend[0, 0]),
        int(legend[1, 0]): int(legend[1, 1]),
        int(legend[1, 1]): int(legend[1, 0]),
    }


def _complete_broken_legend(patch: np.ndarray, bg: int, palette: set[int]) -> np.ndarray:
    assert patch.shape == (2, 2)
    holes = np.argwhere(patch == bg)
    assert holes.shape == (1, 2), "broken legend must have exactly one background hole"
    present = set(map(int, patch[patch != bg].ravel()))
    missing = list(palette - present)
    assert len(missing) == 1, "missing color must be uniquely inferable from the body palette"
    out = patch.copy()
    hr, hc = map(int, holes[0])
    out[hr, hc] = missing[0]
    return out


def _apply_map_on_coords(grid: np.ndarray, coords: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for src, dst in mapping.items():
        mask = np.zeros(grid.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = grid[coords[:, 0], coords[:, 1]] == src
        out[mask] = dst
    return out


def _extract_patch(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _analyze_scene(grid: np.ndarray) -> dict:
    bg = _get_bg(grid)
    comps = _find_components(grid, bg)

    body_candidates = [comp for comp in comps if comp["area"] > 4 and len(comp["colors"]) == 4]
    full_candidates = []
    broken_candidates = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = _extract_patch(grid, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        if comp["area"] == 4 and len(comp["colors"]) == 4:
            full_candidates.append((comp, patch))
        elif comp["area"] == 3 and len(comp["colors"]) == 3 and int(np.count_nonzero(patch == bg)) == 1:
            broken_candidates.append((comp, patch))

    matches = []
    for body in body_candidates:
        palette = set(body["colors"])
        for full_comp, full_patch in full_candidates:
            if set(map(int, full_patch.ravel())) != palette:
                continue
            for broken_comp, broken_patch in broken_candidates:
                if not set(map(int, broken_patch[broken_patch != bg].ravel())).issubset(palette):
                    continue
                true_legend = _complete_broken_legend(broken_patch, bg, palette)
                if set(map(int, true_legend.ravel())) != palette:
                    continue
                if np.array_equal(true_legend, full_patch):
                    continue
                visible = broken_patch != bg
                if not np.any(full_patch[visible] != broken_patch[visible]):
                    continue
                if _diag_map(full_patch) == _diag_map(true_legend):
                    continue
                matches.append(
                    {
                        "bg": bg,
                        "body": body,
                        "full": full_comp,
                        "full_patch": full_patch,
                        "broken": broken_comp,
                        "broken_patch": broken_patch,
                        "true_legend": true_legend,
                    }
                )

    assert len(matches) == 1, "expected a unique large body plus one full and one broken 2x2 legend pair"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Use the broken 2x2 block as the true legend, complete it, then swap body colors across its diagonals.

    Input structure:
    - A dominant background separates one large 4-connected multicolor body from two
      separate 2x2 blocks.
    - One 2x2 block is complete and uses the same four colors as the body.
    - The other 2x2 block shows only three of those colors and leaves one cell as a
      background hole.

    Rule:
    - Ignore the complete 2x2 block.
    - Treat the broken 2x2 block as the true legend: fill its hole with the unique
      body color missing from its three visible cells.
    - Recolor only the large body, simultaneously mapping each body color to the
      color in the diagonally opposite cell of that completed 2x2 legend.
    - Leave both 2x2 blocks and the background unchanged.

    Key insight:
    - The visually complete 2x2 block is a decoy; only the incomplete block reveals
      the real legend because its missing color must be inferred before reading the
      diagonal swap pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    scene = _analyze_scene(g)
    mapping = _diag_map(scene["true_legend"])
    return _apply_map_on_coords(g, scene["body"]["coords"], mapping)


# --------------- generator ---------------
_GEN_GRID_H = (12, 18)
_GEN_GRID_W = (12, 18)
_GEN_BODY_H = range(5, 9)
_GEN_BODY_W = range(5, 9)
_GEN_BODY_AREA = (16, 28)
_GEN_DENSITY = (0.45, 0.75)
_GEN_MIN_PER_COLOR = 2
_GEN_MAX_ATTEMPTS = 600
_GEN_BBOX_GAP = 1


def _gen_valid_dims(area: int) -> list[tuple[int, int]]:
    dims = []
    for h in _GEN_BODY_H:
        for w in _GEN_BODY_W:
            density = area / (h * w)
            if _GEN_DENSITY[0] <= density <= _GEN_DENSITY[1] and h + w - 1 <= area:
                dims.append((h, w))
    return dims


def _gen_spanning_mask(h: int, w: int, target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if random.random() < 0.5:
        c = random.randrange(w)
        mask[0, c] = True
        for r in range(1, h):
            prev = c
            step_choices = [0]
            if c > 0:
                step_choices.append(-1)
            if c + 1 < w:
                step_choices.append(1)
            c = prev + random.choice(step_choices)
            mask[r, prev] = True
            mask[r, c] = True
        mask[random.randrange(1, h - 1), :] = True
    else:
        r = random.randrange(h)
        mask[r, 0] = True
        for c in range(1, w):
            prev = r
            step_choices = [0]
            if r > 0:
                step_choices.append(-1)
            if r + 1 < h:
                step_choices.append(1)
            r = prev + random.choice(step_choices)
            mask[prev, c] = True
            mask[r, c] = True
        mask[:, random.randrange(1, w - 1)] = True

    while int(mask.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _gen_body_patch(palette: list[int], bg: int) -> np.ndarray | None:
    area = random.randint(*_GEN_BODY_AREA)
    dims = _gen_valid_dims(area)
    if not dims:
        return None
    h, w = random.choice(dims)
    mask = _gen_spanning_mask(h, w, area)
    if int(mask.sum()) != area or not _is_contiguous(mask):
        return None
    if _bbox_from_coords([tuple(map(int, rc)) for rc in np.argwhere(mask)]) != (0, 0, h - 1, w - 1):
        return None

    counts = [_GEN_MIN_PER_COLOR] * 4
    remaining = area - sum(counts)
    while remaining > 0:
        idx = random.randrange(4)
        counts[idx] += 1
        remaining -= 1

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    values: list[int] = []
    for color, count in zip(palette, counts):
        values.extend([color] * count)
    random.shuffle(values)

    patch = np.full((h, w), bg, dtype=int)
    for (r, c), color in zip(coords, values):
        patch[r, c] = color
    return patch


def _gen_true_and_decoy_legends(palette: list[int], bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    perms = list(itertools.permutations(palette, 4))
    random.shuffle(perms)
    for true_perm in perms:
        true_legend = np.array(true_perm, dtype=int).reshape(2, 2)
        hole_idx = random.randrange(4)
        broken = true_legend.copy()
        broken.flat[hole_idx] = bg
        for full_perm in perms:
            full = np.array(full_perm, dtype=int).reshape(2, 2)
            if np.array_equal(full, true_legend):
                continue
            visible = broken != bg
            if not np.any(full[visible] != broken[visible]):
                continue
            if _diag_map(full) == _diag_map(true_legend):
                continue
            return true_legend, broken, full
    return None


def _separated(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        box_a[2] + gap < box_b[0]
        or box_b[2] + gap < box_a[0]
        or box_a[3] + gap < box_b[1]
        or box_b[3] + gap < box_a[1]
    )


def _choose_positions(grid_h: int, grid_w: int, shapes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    placed_boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int]] = []
    for h, w in shapes:
        candidates = []
        for r in range(grid_h - h + 1):
            for c in range(grid_w - w + 1):
                box = (r, c, r + h - 1, c + w - 1)
                if all(_separated(box, other, _GEN_BBOX_GAP) for other in placed_boxes):
                    candidates.append((r, c, box))
        if not candidates:
            return None
        r, c, box = random.choice(candidates)
        positions.append((r, c))
        placed_boxes.append(box)
    return positions


def _wrong_output_variants(grid: np.ndarray, scene: dict) -> list[np.ndarray]:
    body_coords = scene["body"]["coords"]
    true_legend = scene["true_legend"]
    full_patch = scene["full_patch"]
    return [
        _apply_map_on_coords(grid, body_coords, _diag_map(full_patch)),
        _apply_map_on_coords(grid, body_coords, _row_map(true_legend)),
        _apply_map_on_coords(grid, body_coords, _row_map(full_patch)),
    ]


def _body_coords_from_patch(patch: np.ndarray, top: int, left: int, bg: int) -> np.ndarray:
    coords = np.argwhere(patch != bg).copy()
    coords[:, 0] += top
    coords[:, 1] += left
    return coords


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*_GEN_GRID_H)
    grid_w = random.randint(*_GEN_GRID_W)
    bg = random.randrange(10)
    palette = random.sample([c for c in range(10) if c != bg], 4)

    body = _gen_body_patch(palette, bg)
    legends = _gen_true_and_decoy_legends(palette, bg)
    if body is None or legends is None:
        return None
    true_legend, broken, full = legends

    order = [body.shape, full.shape, broken.shape]
    positions = _choose_positions(grid_h, grid_w, order)
    if positions is None:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for patch, (r, c) in zip((body, full, broken), positions):
        _paste(grid, patch, r, c, bg)

    expected = _apply_map_on_coords(
        grid,
        _body_coords_from_patch(body, positions[0][0], positions[0][1], bg),
        _diag_map(true_legend),
    )

    try:
        scene = _analyze_scene(grid)
        solved = main(grid)
    except AssertionError:
        return None

    if len(_find_components(grid, bg)) != 3:
        return None
    if scene["bg"] != bg:
        return None
    if set(scene["body"]["colors"]) != set(palette):
        return None
    if not np.array_equal(scene["true_legend"], true_legend):
        return None
    if int(np.count_nonzero(solved != grid)) == 0:
        return None
    if not np.array_equal(solved, expected):
        return None

    for wrong in _wrong_output_variants(grid, scene):
        if np.array_equal(wrong, solved):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {_GEN_MAX_ATTEMPTS} attempts")
