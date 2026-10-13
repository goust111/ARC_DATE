import random
from collections import Counter

import numpy as np


BACKGROUND = 0
TEMPLATE_COLORS = (1, 3, 4)
MARKER = 2
FRAME = 5
MAX_ATTEMPTS = 400


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _is_rect_frame(coords):
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    if set(coords) != expected:
        return None
    return r0, c0, r1, c1


def _parse_grid(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template_mask = ~np.isin(g, [BACKGROUND, MARKER, FRAME])
    template_comps = _components(template_mask)
    assert len(template_comps) == 1, "expected one multicolor template"
    tr0, tc0, tr1, tc1 = _bbox(template_comps[0])
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    assert template.shape in ((3, 4), (4, 3)), "template must be 3x4 or 4x3"
    assert np.all(template != BACKGROUND), "template must be solid"
    assert set(np.unique(template)) == set(TEMPLATE_COLORS), "template must use exactly blue, green, yellow"
    assert int(np.sum(template_mask)) == template.size, "template bbox must contain no extra colors"

    frames = []
    for comp in _components(g == FRAME):
        bbox = _is_rect_frame(comp)
        assert bbox is not None, "gray objects must be one-cell-thick hollow rectangles"
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        frames.append(
            {
                "bbox": bbox,
                "interior_shape": tuple(interior.shape),
                "marker_count": int(np.count_nonzero(interior == MARKER)),
            }
        )
    assert len(frames) == 3, "expected exactly three frames"
    return {
        "grid": g,
        "template_bbox": (tr0, tc0, tr1, tc1),
        "template": template,
        "frames": frames,
    }


def _target_frame(parsed):
    counts = [frame["marker_count"] for frame in parsed["frames"]]
    best = max(counts)
    winners = [i for i, count in enumerate(counts) if count == best]
    assert len(winners) == 1, "marker counts must have a unique maximum"
    return parsed["frames"][winners[0]]


def _oriented_template(template, interior_shape):
    if template.shape == interior_shape:
        return template
    if template.shape[::-1] == interior_shape:
        return np.rot90(template, k=-1)
    raise AssertionError("target frame interior must match the template or its 90-degree rotation")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the uniquely most-marked frame with the corner template after the rotation that matches its interior size.

    Input structure:
    - The background is black.
    - The grid contains exactly three gray one-cell-thick hollow rectangular frames, with red marker pixels scattered inside them.
    - A solid multicolor 3x4 or 4x3 template made of blue, green, and yellow cells sits in one corner and is separate from every frame.

    Rule:
    - Count the red marker pixels inside each gray frame and select the unique frame with the largest count.
    - Compare that frame's interior size with the template size: use the template as-is if the dimensions match, otherwise rotate it 90 degrees clockwise if the dimensions are swapped.
    - Overwrite only the selected frame interior with the oriented template; preserve the template itself, every gray border, and all non-selected frames unchanged.

    Key insight:
    - The red counts decide which frame to modify, while the selected frame's interior geometry independently decides whether the template must stay put or rotate.
    """
    parsed = _parse_grid(grid)
    target = _target_frame(parsed)
    oriented = _oriented_template(parsed["template"], target["interior_shape"])

    out = parsed["grid"].copy()
    r0, c0, r1, c1 = target["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = oriented
    return out


def _random_template(shape):
    while True:
        template = np.random.choice(TEMPLATE_COLORS, size=shape)
        if set(np.unique(template)) != set(TEMPLATE_COLORS):
            continue
        if len({tuple(row) for row in template}) < 2:
            continue
        if len({tuple(col) for col in template.T}) < 2:
            continue
        changes = 0
        changes += int(np.sum(template[:, 1:] != template[:, :-1]))
        changes += int(np.sum(template[1:, :] != template[:-1, :]))
        if changes < template.size // 2:
            continue
        return template.astype(int)


def _rects_separated(a, b, gap=1):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _frame_patch(interior_shape, marker_count):
    ih, iw = interior_shape
    patch = np.full((ih + 2, iw + 2), BACKGROUND, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    cells = [(r, c) for r in range(1, ih + 1) for c in range(1, iw + 1)]
    for r, c in random.sample(cells, marker_count):
        patch[r, c] = MARKER
    return patch


def _paste(canvas, patch, r0, c0):
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _corner_position(grid_shape, patch_shape, corner):
    gh, gw = grid_shape
    ph, pw = patch_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, gw - pw
    if corner == "bl":
        return gh - ph, 0
    if corner == "br":
        return gh - ph, gw - pw
    raise ValueError(corner)


def _choose_counts(winner_index, capacities):
    cap_min = min(capacities)
    target_count = random.randint(max(3, cap_min // 3), min(cap_min, 8))
    pool = list(range(1, target_count))
    a, b = random.sample(pool, 2)
    counts = [a, b, b]
    other = [i for i in range(3) if i != winner_index]
    counts[other[0]] = a
    counts[other[1]] = b
    counts[winner_index] = target_count
    assert len(set(counts)) == 3
    return counts


def _attempt_generate():
    template_shape = random.choice([(3, 4), (4, 3)])
    template = _random_template(template_shape)
    winner_uses_same_orientation = random.choice([False, True])
    frame_shapes = [template_shape, template_shape[::-1], (4, 4)]
    winner_index = 0 if winner_uses_same_orientation else 1

    capacities = [shape[0] * shape[1] for shape in frame_shapes]
    counts = _choose_counts(winner_index, capacities)
    frame_patches = [_frame_patch(shape, count) for shape, count in zip(frame_shapes, counts)]

    gh = random.randint(18, 24)
    gw = random.randint(18, 24)
    grid = np.zeros((gh, gw), dtype=int)

    corner = random.choice(["tl", "tr", "bl", "br"])
    tr0, tc0 = _corner_position(grid.shape, template.shape, corner)
    _paste(grid, template, tr0, tc0)
    placed_boxes = [(tr0, tc0, tr0 + template.shape[0] - 1, tc0 + template.shape[1] - 1)]

    order = list(range(3))
    random.shuffle(order)
    positions = [None, None, None]
    for idx in sorted(order, key=lambda i: frame_patches[i].shape[0] * frame_patches[i].shape[1], reverse=True):
        patch = frame_patches[idx]
        ph, pw = patch.shape
        candidates = []
        for r0 in range(1, gh - ph):
            for c0 in range(1, gw - pw):
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if all(_rects_separated(box, other, gap=1) for other in placed_boxes):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (r0, c0)
        placed_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
        _paste(grid, patch, r0, c0)

    if positions[winner_index] is None:
        return None
    if not _semantic_check(grid):
        return None
    return grid


def _semantic_check(grid):
    parsed = _parse_grid(grid)
    template_shape = parsed["template"].shape
    compatible = [
        i
        for i, frame in enumerate(parsed["frames"])
        if frame["interior_shape"] in (template_shape, template_shape[::-1])
    ]
    if len(compatible) != 2:
        return False

    counts = [frame["marker_count"] for frame in parsed["frames"]]
    if len(set(counts)) != 3:
        return False

    winner = _target_frame(parsed)
    if winner["interior_shape"] not in (template_shape, template_shape[::-1]):
        return False

    out = main(grid)
    changed = np.argwhere(out != grid)
    if changed.size == 0:
        return False

    r0, c0, r1, c1 = winner["bbox"]
    change_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    for rr, cc in changed:
        if not (change_box[0] <= rr <= change_box[2] and change_box[1] <= cc <= change_box[3]):
            return False

    expected = _oriented_template(parsed["template"], winner["interior_shape"])
    if not np.array_equal(out[r0 + 1 : r1, c0 + 1 : c1], expected):
        return False

    other_compatible = [i for i in compatible if parsed["frames"][i]["bbox"] != winner["bbox"]][0]
    alt = grid.copy()
    ar0, ac0, ar1, ac1 = parsed["frames"][other_compatible]["bbox"]
    alt_oriented = _oriented_template(parsed["template"], parsed["frames"][other_compatible]["interior_shape"])
    alt[ar0 + 1 : ar1, ac0 + 1 : ac1] = alt_oriented
    if np.array_equal(alt, out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marked-frame rotation-fill puzzle")
