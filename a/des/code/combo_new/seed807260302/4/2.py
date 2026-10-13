from __future__ import annotations

from collections import Counter
import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return (tuple(map(int, m.shape)), m.tobytes())


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), -int(turns)).copy()


def _components_bool(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == bg or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(len(coords)),
                    "color": color,
                }
            )
    return comps


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_holes(template_patch: np.ndarray, bg: int) -> list[dict]:
    bg_mask = template_patch == bg
    h, w = bg_mask.shape
    reachable = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                stack.append((nr, nc))

    holes: list[dict] = []
    for coords in _components_bool(bg_mask & ~reachable):
        r0, c0, r1, c1 = _bbox(coords)
        rel = coords - np.array([r0, c0])
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append({"origin": (r0, c0), "mask": mask, "key": _mask_key(mask)})
    return holes


def _quadrant_turns(coords: np.ndarray, shape: tuple[int, int]) -> int:
    h, w = shape
    center_r = float(coords[:, 0].mean())
    center_c = float(coords[:, 1].mean())
    top = center_r < h / 2.0
    left = center_c < w / 2.0
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and not left:
        return 2
    return 3


def main(grid: np.ndarray) -> np.ndarray:
    """Embed scattered monochrome fragments into matching holes of a central template using quadrant-controlled rotation.

    Input structure:
    - The background is black/dominant background color.
    - A large central monochrome grey template component contains several enclosed background-color holes.
    - Outside the template, monochrome 4-connected fragments are scattered in the four grid quadrants.

    Rule:
    - Identify the large template component and its enclosed background holes.
    - For each outside fragment, rotate its tight shape by the angle specified by its quadrant: upper-left 0 degrees, upper-right 90 degrees clockwise, lower-right 180 degrees, and lower-left 270 degrees clockwise.
    - The rotated fragment shape is matched exactly to one template hole, that hole is filled with the fragment color, and the original outside fragment cells are removed to background.

    Key insight:
    - Shape matching selects the destination hole only after applying the hidden quadrant-to-rotation rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)

    comps = _color_components(g, bg)
    assert comps, "expected a template and outside fragments"
    max_area = max(comp["area"] for comp in comps)
    template_candidates = [i for i, comp in enumerate(comps) if comp["area"] == max_area]
    assert len(template_candidates) == 1, "template must be the unique largest component"
    template_idx = template_candidates[0]
    template = comps[template_idx]
    tr0, tc0, tr1, tc1 = template["bbox"]

    holes = _find_holes(g[tr0 : tr1 + 1, tc0 : tc1 + 1], bg)
    assert holes, "template must contain holes"
    holes_by_key: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for i, hole in enumerate(holes):
        holes_by_key.setdefault(hole["key"], []).append(i)

    out = g.copy()
    for i, comp in enumerate(comps):
        if i == template_idx:
            continue
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    used: set[int] = set()
    for i, comp in enumerate(comps):
        if i == template_idx:
            continue
        coords = comp["coords"]
        turns = _quadrant_turns(coords, g.shape)
        rotated = _rotate_cw(_tight_mask(coords), turns)
        matches = [idx for idx in holes_by_key.get(_mask_key(rotated), []) if idx not in used]
        assert len(matches) == 1, "each rotated fragment shape must match exactly one unused hole"
        hole_idx = matches[0]
        used.add(hole_idx)

        hr0, hc0 = holes[hole_idx]["origin"]
        mask = holes[hole_idx]["mask"]
        rr = tr0 + hr0
        cc = tc0 + hc0
        region = out[rr : rr + mask.shape[0], cc : cc + mask.shape[1]]
        region[mask] = int(comp["color"])

    assert len(used) == len(holes), "every template hole must be filled"
    return out


# ---------------- generator ----------------

GEN_BG = 0
GEN_TEMPLATE = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_GRID_RANGE = (27, 30)
GEN_TEMPLATE_RANGE = (10, 13)
GEN_PIECE_COUNT_RANGE = (4, 7)
GEN_MIN_MARGIN = 6
GEN_PADDING = 1
GEN_ATTEMPTS = 400


def _shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[0, 1, 1], [1, 1, 0], [1, 0, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]],
        [[0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    ]
    shapes = []
    canonical_seen: set[tuple[tuple[int, int], bytes]] = set()
    for rows in raw:
        mask = np.array(rows, dtype=bool)
        if len(_components_bool(mask)) != 1:
            continue
        rotations = [_mask_key(_rotate_cw(mask, k)) for k in range(4)]
        if len(set(rotations)) != 4:
            continue
        canonical = min(rotations)
        if canonical in canonical_seen:
            continue
        canonical_seen.add(canonical)
        shapes.append(mask)
    return shapes


SHAPES = _shape_library()


def _can_place_hole(holes: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> bool:
    h, w = holes.shape
    sh, sw = shape.shape
    if r0 < 1 or c0 < 1 or r0 + sh > h - 1 or c0 + sw > w - 1:
        return False
    if holes[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
        return False
    return True


def _place_holes(th: int, tw: int, shapes: list[np.ndarray]) -> list[tuple[np.ndarray, int, int]] | None:
    hole_cells = np.zeros((th, tw), dtype=bool)
    placed: list[tuple[np.ndarray, int, int]] = []
    for shape in shapes:
        sh, sw = shape.shape
        options = [
            (r, c)
            for r in range(1, th - sh)
            for c in range(1, tw - sw)
            if _can_place_hole(hole_cells, shape, r, c)
        ]
        if not options:
            return None
        r0, c0 = random.choice(options)
        hole_cells[r0 : r0 + sh, c0 : c0 + sw][shape] = True
        placed.append((shape, r0, c0))
    return placed


def _quad_box(q: int, fr: int, fc: int, th: int, tw: int, h: int, w: int) -> tuple[int, int, int, int]:
    if q == 0:
        return (1, 1, fr - 2, fc - 2)
    if q == 1:
        return (1, fc + tw + 2, fr - 2, w - 2)
    if q == 2:
        return (fr + th + 2, fc + tw + 2, h - 2, w - 2)
    return (fr + th + 2, 1, h - 2, fc - 2)


def _can_place_piece(occupied: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    sh, sw = shape.shape
    if r0 < 0 or c0 < 0 or r0 + sh > h or c0 + sw > w:
        return False
    return not occupied[
        max(0, r0 - GEN_PADDING) : min(h, r0 + sh + GEN_PADDING),
        max(0, c0 - GEN_PADDING) : min(w, c0 + sw + GEN_PADDING),
    ].any()


def _place_piece(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: np.ndarray,
    color: int,
    box: tuple[int, int, int, int],
) -> bool:
    rlo, clo, rhi, chi = box
    sh, sw = shape.shape
    if rhi - rlo + 1 < sh or chi - clo + 1 < sw:
        return False
    options = [
        (r, c)
        for r in range(rlo, rhi - sh + 2)
        for c in range(clo, chi - sw + 2)
        if _can_place_piece(occupied, shape, r, c)
    ]
    if not options:
        return False
    r0, c0 = random.choice(options)
    grid[r0 : r0 + sh, c0 : c0 + sw][shape] = color
    occupied[
        max(0, r0 - GEN_PADDING) : min(grid.shape[0], r0 + sh + GEN_PADDING),
        max(0, c0 - GEN_PADDING) : min(grid.shape[1], c0 + sw + GEN_PADDING),
    ] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_RANGE)
    w = random.randint(*GEN_GRID_RANGE)
    th = random.randint(*GEN_TEMPLATE_RANGE)
    tw = random.randint(*GEN_TEMPLATE_RANGE)
    if h - th < 2 * GEN_MIN_MARGIN or w - tw < 2 * GEN_MIN_MARGIN:
        return None
    fr = random.randint(GEN_MIN_MARGIN, h - th - GEN_MIN_MARGIN)
    fc = random.randint(GEN_MIN_MARGIN, w - tw - GEN_MIN_MARGIN)

    n = random.randint(*GEN_PIECE_COUNT_RANGE)
    if n > len(SHAPES):
        return None
    base_shapes = random.sample(SHAPES, n)
    hole_shapes = [_rotate_cw(shape, random.randrange(4)) for shape in base_shapes]
    hole_specs = _place_holes(th, tw, hole_shapes)
    if hole_specs is None:
        return None

    colors = random.sample(GEN_COLORS, n)
    quadrants = [0, 1, 2, 3] + [random.randrange(4) for _ in range(n - 4)]
    random.shuffle(quadrants)

    inp = np.full((h, w), GEN_BG, dtype=int)
    expected = np.full((h, w), GEN_BG, dtype=int)
    inp[fr : fr + th, fc : fc + tw] = GEN_TEMPLATE
    expected[fr : fr + th, fc : fc + tw] = GEN_TEMPLATE

    for (shape, r0, c0), color in zip(hole_specs, colors):
        inp[fr + r0 : fr + r0 + shape.shape[0], fc + c0 : fc + c0 + shape.shape[1]][shape] = GEN_BG
        expected[fr + r0 : fr + r0 + shape.shape[0], fc + c0 : fc + c0 + shape.shape[1]][shape] = color

    occupied = np.zeros((h, w), dtype=bool)
    occupied[
        max(0, fr - GEN_PADDING) : min(h, fr + th + GEN_PADDING),
        max(0, fc - GEN_PADDING) : min(w, fc + tw + GEN_PADDING),
    ] = True

    tasks = list(zip(hole_specs, colors, quadrants))
    random.shuffle(tasks)
    for (shape, _r0, _c0), color, q in tasks:
        fragment = _rotate_cw(shape, (4 - q) % 4)
        if not _place_piece(inp, occupied, fragment, color, _quad_box(q, fr, fc, th, tw, h, w)):
            return None

    if _background(inp) != GEN_BG:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-rotation hole-fill input")
