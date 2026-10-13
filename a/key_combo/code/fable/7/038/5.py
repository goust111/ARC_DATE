"""ARC task: quadrant-oriented pinwheel stamping."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1)


def _bbox(coords: np.ndarray) -> BBox:
    """Return the inclusive bbox of (row, column) coordinates."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _crop(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _foreground_components(grid: np.ndarray) -> list[dict]:
    """Find 8-connected, potentially multicolored, non-black objects."""
    foreground = grid != 0
    seen = np.zeros(grid.shape, dtype=bool)
    objects: list[dict] = []
    height, width = grid.shape
    for start_r, start_c in np.argwhere(foreground):
        start_r, start_c = int(start_r), int(start_c)
        if seen[start_r, start_c]:
            continue
        seen[start_r, start_c] = True
        queue = deque([(start_r, start_c)])
        coords: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < height and 0 <= cc < width and
                            foreground[rr, cc] and not seen[rr, cc]):
                        seen[rr, cc] = True
                        queue.append((rr, cc))
        array = np.asarray(coords, dtype=int)
        colors = set(map(int, grid[array[:, 0], array[:, 1]]))
        objects.append({"coords": array, "bbox": _bbox(array), "colors": colors})
    return objects


def _extract_sprites(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    objects = _foreground_components(grid)
    assert len(objects) == 2, "expected exactly two separated foreground sprites"
    templates = [obj for obj in objects if len(obj["colors"]) in (2, 3)]
    stamps = [obj for obj in objects if len(obj["colors"]) == 1]
    assert len(templates) == 1 and len(stamps) == 1, (
        "one sprite must be a 2-3 color template and the other monochrome"
    )
    template = _crop(grid, templates[0]["bbox"])
    stamp = _crop(grid, stamps[0]["bbox"])
    return template, stamp


def _quarter_turn(row: int, col: int, height: int, width: int) -> int:
    """Clockwise quarter-turns for TL, TR, BR, BL pinwheel quadrants."""
    if row < height // 2:
        return 0 if col < width // 2 else 1
    return 3 if col < width // 2 else 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a recolored asymmetric sprite through a multicolor template, rotating copies by quadrant.

    Input structure:
    - A black grid contains exactly two well-separated sprites: one 2-3 color
      template with even height and width, and one monochrome asymmetric stamp.
    - The stamp has a square bounding box; black cells inside either bounding
      box are transparent background, and the two sprite sizes may differ.

    Rule:
    - Make one stamp-sized output block for every template cell.  A black
      template cell gives an all-black block; a colored cell receives the
      stamp mask recolored to that cell's color.
    - Rotate the mask clockwise by 0, 90, 180, or 270 degrees when the cell is
      respectively in the template's upper-left, upper-right, lower-right, or
      lower-left quadrant.

    Key insight:
    - The multicolor sprite supplies block positions and colors, while the
      monochrome sprite supplies shape; position in the template, not color,
      supplies each copy's pinwheel orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    template, stamp = _extract_sprites(g)
    template_h, template_w = template.shape
    stamp_h, stamp_w = stamp.shape
    assert template_h % 2 == 0 and template_w % 2 == 0, (
        "template dimensions must both be even"
    )
    assert stamp_h == stamp_w, "the rotating stamp must have a square bbox"
    assert len(set(map(int, stamp[stamp != 0]))) == 1, "stamp must be monochrome"
    stamp_mask = stamp != 0
    rotations = [np.rot90(stamp_mask, -turn) for turn in range(4)]

    out = np.zeros((template_h * stamp_h, template_w * stamp_w), dtype=int)
    assert max(out.shape) <= 30, "output exceeds ARC's 30x30 size limit"
    for row in range(template_h):
        for col in range(template_w):
            color = int(template[row, col])
            if color == 0:
                continue
            turn = _quarter_turn(row, col, template_h, template_w)
            r0, c0 = row * stamp_h, col * stamp_w
            block = out[r0 : r0 + stamp_h, c0 : c0 + stamp_w]
            block[rotations[turn]] = color
    return out


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    reached = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if (0 <= nxt[0] < mask.shape[0] and 0 <= nxt[1] < mask.shape[1]
                    and mask[nxt] and nxt not in reached):
                reached.add(nxt)
                queue.append(nxt)
    return len(reached) == int(mask.sum())


def _make_stamp(side: int) -> np.ndarray:
    """Generate a 4-connected mask with full bbox and four distinct rotations."""
    for _ in range(500):
        mask = np.random.random((side, side)) < random.uniform(0.38, 0.68)
        if not (4 <= int(mask.sum()) <= side * side - 2):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _connected4(mask):
            continue
        turns = [np.rot90(mask, k) for k in range(4)]
        if all(not np.array_equal(turns[i], turns[j]) for i in range(4) for j in range(i)):
            return mask
    raise RuntimeError("failed to make an asymmetric stamp")


def _make_template(height: int, width: int, colors: list[int]) -> np.ndarray:
    """Make a connected, holed mask whose colors recur across quadrants."""
    for _ in range(500):
        mask = np.random.random((height, width)) < random.uniform(0.62, 0.82)
        # Occupied edges make the visible bbox equal the intended even dimensions.
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if mask.all() or not _connected4(mask):
            continue
        quadrants = [
            (slice(0, height // 2), slice(0, width // 2)),
            (slice(0, height // 2), slice(width // 2, width)),
            (slice(height // 2, height), slice(width // 2, width)),
            (slice(height // 2, height), slice(0, width // 2)),
        ]
        if any(int(mask[q].sum()) < len(colors) for q in quadrants):
            continue

        template = np.zeros((height, width), dtype=int)
        for q in quadrants:
            positions = np.argwhere(mask[q])
            positions = positions[np.random.permutation(len(positions))]
            r_off = q[0].start or 0
            c_off = q[1].start or 0
            # Every color appears in every quadrant.  Thus color cannot encode turn.
            for color, (r, c) in zip(colors, positions[:len(colors)]):
                template[int(r) + r_off, int(c) + c_off] = color
        remaining = np.argwhere(mask & (template == 0))
        for r, c in remaining:
            template[int(r), int(c)] = random.choice(colors)
        return template
    raise RuntimeError("failed to make a valid template")


def _boxes_far(a: BBox, b: BBox, gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (ar1 + gap < br0 or br1 + gap < ar0 or
            ac1 + gap < bc0 or bc1 + gap < ac0)


def _independent_oracle(template: np.ndarray, stamp_mask: np.ndarray) -> np.ndarray:
    """Coordinate-form oracle, intentionally separate from main's rot90 overlay."""
    th, tw = template.shape
    side = stamp_mask.shape[0]
    out = np.zeros((th * side, tw * side), dtype=int)
    points = [tuple(map(int, point)) for point in np.argwhere(stamp_mask)]
    for tr in range(th):
        for tc in range(tw):
            color = int(template[tr, tc])
            if color == 0:
                continue
            turn = _quarter_turn(tr, tc, th, tw)
            for r, c in points:
                if turn == 0:
                    rr, cc = r, c
                elif turn == 1:
                    rr, cc = c, side - 1 - r
                elif turn == 2:
                    rr, cc = side - 1 - r, side - 1 - c
                else:
                    rr, cc = side - 1 - c, r
                out[tr * side + rr, tc * side + cc] = color
    return out


def _validate_generated(grid: np.ndarray, template: np.ndarray, stamp_mask: np.ndarray) -> bool:
    """Check uniqueness, stated semantics, and resistance to simple shortcuts."""
    if max(grid.shape) > 30 or len(_foreground_components(grid)) != 2:
        return False
    try:
        found_template, found_stamp = _extract_sprites(grid)
        answer = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(found_template, template) or not np.array_equal(found_stamp != 0, stamp_mask):
        return False
    if not np.array_equal(answer, _independent_oracle(template, stamp_mask)):
        return False

    # All four orientation branches are visible and distinct.
    if any(not np.any(template[q]) for q in (
        np.s_[:template.shape[0] // 2, :template.shape[1] // 2],
        np.s_[:template.shape[0] // 2, template.shape[1] // 2:],
        np.s_[template.shape[0] // 2:, template.shape[1] // 2:],
        np.s_[template.shape[0] // 2:, :template.shape[1] // 2],
    )):
        return False
    # Explicitly construct the common wrong rule: stamp every block unrotated.
    side = stamp_mask.shape[0]
    shortcut = np.zeros_like(answer)
    for r, c in np.argwhere(template != 0):
        block = shortcut[r * side:(r + 1) * side, c * side:(c + 1) * side]
        block[stamp_mask] = int(template[r, c])
    if np.array_equal(answer, shortcut):
        return False
    # Black holes make selection of template positions necessary; multiple colors
    # make per-cell recoloring necessary.  A non-solid stamp makes its shape necessary.
    return bool(
        np.any(template == 0)
        and len(set(map(int, template[template != 0]))) >= 2
        and np.any(~stamp_mask)
    )


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate instances satisfying all solver assumptions."""
    for _ in range(1000):
        template_h, template_w = random.choice([(4, 4), (4, 6), (6, 4), (6, 6)])
        stamp_side = random.choice([3, 4])
        if max(template_h * stamp_side, template_w * stamp_side) > 30:
            continue
        template_colors = random.sample(range(1, 10), random.choice([2, 3]))
        stamp_color = random.choice([c for c in range(1, 10) if c not in template_colors])
        template = _make_template(template_h, template_w, template_colors)
        stamp_mask = _make_stamp(stamp_side)
        stamp = stamp_mask.astype(int) * stamp_color

        grid_h = random.randint(max(template_h, stamp_side) + 5, 24)
        grid_w = random.randint(max(template_w, stamp_side) + 5, 24)
        for _place in range(200):
            tr = random.randint(0, grid_h - template_h)
            tc = random.randint(0, grid_w - template_w)
            sr = random.randint(0, grid_h - stamp_side)
            sc = random.randint(0, grid_w - stamp_side)
            template_box = (tr, tc, tr + template_h - 1, tc + template_w - 1)
            stamp_box = (sr, sc, sr + stamp_side - 1, sc + stamp_side - 1)
            if not _boxes_far(template_box, stamp_box):
                continue
            grid = np.zeros((grid_h, grid_w), dtype=int)
            grid[tr:tr + template_h, tc:tc + template_w] = template
            grid[sr:sr + stamp_side, sc:sc + stamp_side] = stamp
            if _validate_generated(grid, template, stamp_mask):
                return grid
    raise RuntimeError("failed to generate a valid pinwheel-stamp input")
