from __future__ import annotations

import random

import numpy as np


_BG = 0
_GEN_RETRIES = 400
_GRID_MAX = 30
_FENCE_COUNT_RANGE = (2, 4)
_FENCE_EXTRA_RANGE = (0, 2)
_TEMPLATE_DIM_RANGE = (2, 5)
_MIN_SHAPE_CELLS = 2
_MAX_SHAPE_FILL_RATIO = 0.75


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_rect_frame(points: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False, (r0, c0, r1, c1)
    expected = 2 * (h + w) - 4
    if len(points) != expected:
        return False, (r0, c0, r1, c1)
    point_set = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_perimeter = r in (r0, r1) or c in (c0, c1)
            if on_perimeter != ((r, c) in point_set):
                return False, (r0, c0, r1, c1)
    return True, (r0, c0, r1, c1)


def _find_fences(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    fences: list[dict[str, object]] = []
    for pts in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        ok, bbox = _is_rect_frame(pts)
        if not ok:
            continue
        r0, c0, r1, c1 = bbox
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_locs = np.argwhere(interior != bg)
        if len(marker_locs) != 1:
            continue
        mr, mc = map(int, marker_locs[0])
        fences.append(
            {
                "frame_color": next(iter(colors)),
                "bbox": bbox,
                "marker_color": int(interior[mr, mc]),
                "marker_pos": (r0 + 1 + mr, c0 + 1 + mc),
            }
        )
    assert fences, "expected at least one fenced rectangle"
    return fences


def _find_templates(
    grid: np.ndarray, bg: int, fences: list[dict[str, object]]
) -> dict[int, tuple[np.ndarray, int]]:
    blocked = np.zeros_like(grid, dtype=bool)
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    templates: dict[int, tuple[np.ndarray, int]] = {}
    comps = _components((grid != bg) & ~blocked)
    for pts in comps:
        colors = {int(grid[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        lr, lc = r0 - 1, c0 - 1
        if lr < 0 or lc < 0 or blocked[lr, lc] or int(grid[lr, lc]) == bg:
            continue
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - r0, c - c0] = True
        label_color = int(grid[lr, lc])
        assert label_color not in templates, "template labels must be unique"
        templates[label_color] = (mask, next(iter(colors)))

    assert templates, "expected at least one external template"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the correct outside template into each fenced interior based on the interior marker color.

    Input structure:
    - The background is the dominant color.
    - Several one-pixel-thick closed rectangular fences appear on the grid; each fence encloses
      background plus exactly one single-pixel colored marker.
    - Outside the fences are several monochrome template blocks, each paired with a single label
      pixel positioned just northwest of the template block's top-left corner.

    Rule:
    - For each fence, read the color of its unique interior marker.
    - Find the outside template whose label pixel has that same color.
    - Remove the marker, copy the template block without its label, and place it inside the fence
      aligned to the fence interior's top-left corner.
    - Clear every outside template block and every label pixel to background.

    Key insight:
    - The interior marker is only a color key: it selects which outside template to import into the
      corresponding fenced region.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    fences = _find_fences(g, bg)
    templates = _find_templates(g, bg, fences)

    out = np.full_like(g, bg)

    for fence in fences:
        frame_color = int(fence["frame_color"])
        r0, c0, r1, c1 = fence["bbox"]
        out[r0 : r1 + 1, c0] = frame_color
        out[r0 : r1 + 1, c1] = frame_color
        out[r0, c0 : c1 + 1] = frame_color
        out[r1, c0 : c1 + 1] = frame_color

        marker_color = int(fence["marker_color"])
        assert marker_color in templates, "every fence marker must match a template label"
        mask, fill_color = templates[marker_color]
        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        th, tw = mask.shape
        assert th <= ih and tw <= iw, "template must fit inside its fence interior"
        region = out[r0 + 1 : r0 + 1 + th, c0 + 1 : c0 + 1 + tw]
        region[mask] = int(fill_color)

    return out


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        h = rng.randint(*_TEMPLATE_DIM_RANGE)
        w = rng.randint(*_TEMPLATE_DIM_RANGE)
        total = h * w
        lo = min(total, max(_MIN_SHAPE_CELLS, total // 3))
        hi = max(lo, int(np.ceil(total * _MAX_SHAPE_FILL_RATIO)))
        target = rng.randint(lo, hi)

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < target and frontier:
            nr, nc = rng.choice(tuple(frontier))
            frontier.remove((nr, nc))
            cells.add((nr, nc))
            for rr, cc in _neighbors4(nr, nc):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))

        if len(cells) < target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        return mask
    raise AssertionError("failed to build a connected template mask")


def _frame_patch(
    interior_shape: tuple[int, int], fence_color: int, marker_color: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    ih, iw = interior_shape
    h = ih + 2
    w = iw + 2
    inp = np.full((h, w), _BG, dtype=int)
    out = np.full((h, w), _BG, dtype=int)
    inp[0, :] = fence_color
    inp[-1, :] = fence_color
    inp[:, 0] = fence_color
    inp[:, -1] = fence_color
    out[:, :] = inp
    mr = rng.randint(1, h - 2)
    mc = rng.randint(1, w - 2)
    inp[mr, mc] = marker_color
    return inp, out


def _template_patch(mask: np.ndarray, label_color: int, fill_color: int) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h + 1, w + 1), _BG, dtype=int)
    patch[0, 0] = label_color
    patch[1:, 1:][mask] = fill_color
    return patch


def _fits(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    gh, gw = used.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(gh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(gw, c0 + w + pad)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _place_bbox(used: np.ndarray, h: int, w: int, rng: random.Random) -> tuple[int, int] | None:
    gh, gw = used.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, gh - h):
        for c0 in range(1, gw - w):
            if _fits(used, r0, c0, h, w):
                candidates.append((r0, c0))
    if not candidates:
        return None
    choice = rng.choice(candidates)
    used[choice[0] : choice[0] + h, choice[1] : choice[1] + w] = True
    return choice


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    block = dst[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != _BG
    block[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_fences = rng.randint(*_FENCE_COUNT_RANGE)
    fence_color = rng.choice(list(range(1, 10)))
    available = [c for c in range(1, 10) if c != fence_color]
    rng.shuffle(available)
    marker_colors = available[:n_fences]
    fill_colors = available[n_fences : 2 * n_fences]

    specs: list[dict[str, object]] = []
    for idx in range(n_fences):
        mask = _random_connected_mask(rng)
        extra_h = rng.randint(*_FENCE_EXTRA_RANGE)
        extra_w = rng.randint(*_FENCE_EXTRA_RANGE)
        interior_shape = (int(mask.shape[0] + extra_h), int(mask.shape[1] + extra_w))
        frame_inp, frame_out = _frame_patch(interior_shape, fence_color, marker_colors[idx], rng)
        template = _template_patch(mask, marker_colors[idx], fill_colors[idx])
        specs.append(
            {
                "mask": mask,
                "fill_color": fill_colors[idx],
                "frame_inp": frame_inp,
                "frame_out": frame_out,
                "template": template,
            }
        )

    min_side = min(_GRID_MAX, max(16, 10 + 4 * n_fences))
    gh = rng.randint(min_side, _GRID_MAX)
    gw = rng.randint(min_side, _GRID_MAX)
    inp = np.full((gh, gw), _BG, dtype=int)
    out = np.full((gh, gw), _BG, dtype=int)
    used = np.zeros((gh, gw), dtype=bool)

    placements: dict[tuple[int, str], tuple[int, int]] = {}
    items: list[tuple[int, int, str, np.ndarray]] = []
    for idx, spec in enumerate(specs):
        frame = spec["frame_inp"]
        template = spec["template"]
        items.append((frame.shape[0] * frame.shape[1], idx, "frame", frame))
        items.append((template.shape[0] * template.shape[1], idx, "template", template))
    items.sort(reverse=True)

    for _area, idx, kind, patch in items:
        pos = _place_bbox(used, patch.shape[0], patch.shape[1], rng)
        if pos is None:
            return None
        placements[(idx, kind)] = pos

    for idx, spec in enumerate(specs):
        fr, fc = placements[(idx, "frame")]
        tr, tc = placements[(idx, "template")]
        _paste(inp, spec["frame_inp"], fr, fc)
        _paste(inp, spec["template"], tr, tc)
        _paste(out, spec["frame_out"], fr, fc)
        mask = spec["mask"]
        fill_color = int(spec["fill_color"])
        out[fr + 1 : fr + 1 + mask.shape[0], fc + 1 : fc + 1 + mask.shape[1]][mask] = fill_color

    if _most_common_color(inp) != _BG:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
