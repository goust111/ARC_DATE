"""ARC task 3: complete a checkerboard-sampled shadow from a colored template."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _background(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    most = counts.most_common()
    assert len(most) == 1 or most[0][1] > most[1][1], "background must be unique"
    return int(most[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords), "expected a nonempty object"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dihedral(array: np.ndarray) -> list[np.ndarray]:
    """The four rotations followed by the four rotations of a reflection."""
    return [np.rot90(array, k).copy() for k in range(4)] + [
        np.rot90(np.fliplr(array), k).copy() for k in range(4)
    ]


def _solutions(grid: np.ndarray) -> list[dict]:
    """Enumerate mask color, D4 pose, local parity, and target alignment."""
    bg = _background(grid)
    h, w = grid.shape
    colors = set(map(int, np.unique(grid))) - {bg}
    answers: list[dict] = []

    for mask_color in colors:
        sparse_coords = np.argwhere(grid == mask_color)
        if len(sparse_coords) < 2:
            continue
        # A checkerboard class cannot contain orthogonally adjacent cells.
        sparse_set = set(map(tuple, sparse_coords.tolist()))
        if any((r + dr, c + dc) in sparse_set for r, c in sparse_set
               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            continue

        template_mask = (grid != bg) & (grid != mask_color)
        template_colors = set(map(int, grid[template_mask]))
        if len(template_colors) < 2:
            continue
        r0, c0, r1, c1 = _bbox(template_mask)
        template = grid[r0:r1 + 1, c0:c1 + 1].copy()
        # The template is one compact sprite; no candidate mask pixels lie in it.
        if np.any(template == mask_color):
            continue
        foreground = template != bg
        if not (np.any(foreground[0]) and np.any(foreground[-1]) and
                np.any(foreground[:, 0]) and np.any(foreground[:, -1])):
            continue

        sr0, sc0 = sparse_coords.min(axis=0)
        sr1, sc1 = sparse_coords.max(axis=0)
        for pose_index, posed in enumerate(_dihedral(template)):
            posed_fg = posed != bg
            ph, pw = posed.shape
            rr, cc = np.indices((ph, pw))
            for parity in (0, 1):
                sampled = posed_fg & ((rr + cc) % 2 == parity)
                local = np.argwhere(sampled)
                if len(local) != len(sparse_coords):
                    continue
                lr0, lc0 = local.min(axis=0)
                lr1, lc1 = local.max(axis=0)
                top = int(sr0 - lr0)
                left = int(sc0 - lc0)
                if int(sr1 - lr1) != top or int(sc1 - lc1) != left:
                    continue
                if top < 0 or left < 0 or top + ph > h or left + pw > w:
                    continue
                shifted = {(top + int(r), left + int(c)) for r, c in local}
                if shifted != sparse_set:
                    continue
                # Apart from the sampled shadow itself, the destination rectangle is clear.
                region = grid[top:top + ph, left:left + pw]
                if np.any((region != bg) & (region != mask_color)):
                    continue
                answers.append({
                    "mask_color": mask_color,
                    "template_bbox": (r0, c0, r1, c1),
                    "pose_index": pose_index,
                    "parity": parity,
                    "top": top,
                    "left": left,
                    "posed": posed,
                })
    return answers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a monochrome checkerboard-sampled shadow with the correctly posed colored template.

    Input structure:
    - A uniquely dominant background separates one compact multicolor template sprite from a
      distant monochrome sparse pattern whose cells are never orthogonally adjacent.
    - The sparse cells occupy one parity class of the foreground of a dihedral pose of the template.

    Rule:
    - Enumerate all eight rotations/reflections of the template, both checkerboard parities, and
      every compatible alignment; the input guarantees exactly one match.
    - At that alignment replace the sparse mask by the complete posed template, copying every
      foreground color while preserving the original template and all other background cells.

    Key insight:
    - The isolated monochrome pixels are only one checkerboard half of a silhouette; their unique
      match determines both the template pose and alignment needed to restore the missing half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    answers = _solutions(g)
    assert len(answers) == 1, f"expected one pose/alignment, found {len(answers)}"
    match = answers[0]
    bg = _background(g)
    posed = match["posed"]
    top, left = match["top"], match["left"]
    out = g.copy()
    region = out[top:top + posed.shape[0], left:left + posed.shape[1]]
    region[region == match["mask_color"]] = bg
    region[posed != bg] = posed[posed != bg]
    return out


def _connected_random_mask(height: int, width: int) -> np.ndarray:
    """Grow an irregular tight-bbox polyomino with holes/indentations."""
    target = random.randint(max(10, (height * width) // 2), height * width - 3)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if not len(coords):
        return False
    seen = {tuple(coords[0])}
    queue = deque(seen)
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p = (r + dr, c + dc)
            if 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and mask[p] and p not in seen:
                seen.add(p)
                queue.append(p)
    return len(seen) == int(mask.sum())


def _place_pair(canvas: np.ndarray, template: np.ndarray, sparse: np.ndarray,
                vertical: bool) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Place tight rectangles with at least a two-cell background corridor."""
    h, w = canvas.shape
    th, tw = template.shape
    sh, sw = sparse.shape
    gap = random.randint(3, 5)
    template_first = random.choice((True, False))
    if vertical:
        if th + sh + gap + 2 > h or max(tw, sw) + 2 > w:
            return None
        first_h, second_h = (th, sh) if template_first else (sh, th)
        r_first = random.randint(1, h - first_h - gap - second_h - 1)
        r_second = random.randint(r_first + first_h + gap, h - second_h - 1)
        p_first = (r_first, random.randint(1, w - ((tw if template_first else sw)) - 1))
        p_second = (r_second, random.randint(1, w - ((sw if template_first else tw)) - 1))
    else:
        if tw + sw + gap + 2 > w or max(th, sh) + 2 > h:
            return None
        first_w, second_w = (tw, sw) if template_first else (sw, tw)
        c_first = random.randint(1, w - first_w - gap - second_w - 1)
        c_second = random.randint(c_first + first_w + gap, w - second_w - 1)
        p_first = (random.randint(1, h - (th if template_first else sh) - 1), c_first)
        p_second = (random.randint(1, h - (sh if template_first else th) - 1), c_second)
    return (p_first, p_second) if template_first else (p_second, p_first)


def _attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    palette = [c for c in range(10) if c != bg]
    mask_color, *rest = random.sample(palette, 5)
    template_colors = rest[:random.choice((3, 4))]

    height = random.randint(4, 7)
    width = random.randint(4, 7)
    if random.random() < 0.5 and height == width:
        width = 4 if width != 4 else 5
    mask = _connected_random_mask(height, width)
    if not _is_connected(mask) or not all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())):
        return None
    # Reject solid/simple silhouettes: completion must infer genuine missing structure.
    if int(mask.sum()) >= height * width - 1:
        return None

    template = np.full((height, width), bg, dtype=int)
    coords = [tuple(x) for x in np.argwhere(mask)]
    assignments = template_colors[:] + [random.choice(template_colors) for _ in range(len(coords) - len(template_colors))]
    random.shuffle(assignments)
    for (r, c), color in zip(coords, assignments):
        template[r, c] = color

    pose_index = random.randrange(8)
    posed = _dihedral(template)[pose_index]
    posed_fg = posed != bg
    rr, cc = np.indices(posed.shape)
    parity = random.randrange(2)
    sampled = posed_fg & ((rr + cc) % 2 == parity)
    omitted = posed_fg & ~sampled
    if int(sampled.sum()) < 5 or int(omitted.sum()) < 4:
        return None
    sparse = np.full(posed.shape, bg, dtype=int)
    sparse[sampled] = mask_color

    canvas_h = random.randint(max(14, height + posed.shape[0] + 6), 25)
    canvas_w = random.randint(max(14, width + posed.shape[1] + 6), 25)
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    placement = _place_pair(canvas, template, sparse, random.choice((True, False)))
    if placement is None:
        return None
    (tr, tc), (sr, sc) = placement
    tview = canvas[tr:tr + height, tc:tc + width]
    tview[template != bg] = template[template != bg]
    sview = canvas[sr:sr + sparse.shape[0], sc:sc + sparse.shape[1]]
    sview[sampled] = mask_color

    matches = _solutions(canvas)
    if len(matches) != 1:
        return None
    match = matches[0]
    if (match["pose_index"], match["parity"], match["top"], match["left"]) != (pose_index, parity, sr, sc):
        return None

    expected = canvas.copy()
    eview = expected[sr:sr + posed.shape[0], sc:sc + posed.shape[1]]
    eview[eview == mask_color] = bg
    eview[posed_fg] = posed[posed_fg]
    try:
        actual = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected) or np.array_equal(actual, canvas):
        return None
    # Semantic checks: sampled evidence, omitted completion, and multicolor transfer all matter.
    if not np.all(canvas[sr:sr + posed.shape[0], sc:sc + posed.shape[1]][sampled] == mask_color):
        return None
    if not np.all(actual[sr:sr + posed.shape[0], sc:sc + posed.shape[1]][omitted] == posed[omitted]):
        return None
    if len(set(map(int, posed[posed_fg]))) < 3:
        return None
    return canvas


def generate_input() -> np.ndarray:
    """Generate varied, uniquely solvable checkerboard-shadow instances."""
    for _ in range(1000):
        result = _attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a unique checkerboard-shadow input")
