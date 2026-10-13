import random
from collections import defaultdict, deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
RECT_DIMS = [(h, w) for h in range(2, 5) for w in range(2, 5)]
LARGE_DIMS = [dims for dims in RECT_DIMS if dims != (2, 2)]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 400


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray, connectivity: int):
    h, w = mask.shape
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr or dc
        )
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_rectangles_and_arrows(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    rectangles = []
    leftover = np.zeros_like(g, dtype=bool)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        mask = g == color
        for coords in _components(mask, connectivity=4):
            r0, c0, r1, c1 = _bbox(coords)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if len(coords) == hh * ww and 2 <= hh <= 4 and 2 <= ww <= 4:
                rectangles.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "coords": coords,
                    }
                )
            else:
                for r, c in coords:
                    leftover[r, c] = True

    arrows = []
    for color in sorted(int(v) for v in np.unique(g[leftover])):
        mask = leftover & (g == color)
        for coords in _components(mask, connectivity=8):
            arrow = _decode_arrow(coords)
            if arrow is None:
                raise AssertionError(f"unrecognized arrow geometry for color {color}: {coords}")
            shaft, direction = arrow
            arrows.append(
                {
                    "color": color,
                    "coords": sorted(coords),
                    "shaft": shaft,
                    "dir": direction,
                }
            )
    return rectangles, arrows


def _decode_arrow(coords):
    if len(coords) != 3:
        return None
    neighbors = {}
    for cell in coords:
        linked = []
        for other in coords:
            if other == cell:
                continue
            dr = other[0] - cell[0]
            dc = other[1] - cell[1]
            if max(abs(dr), abs(dc)) == 1:
                linked.append(other)
        neighbors[cell] = linked

    shafts = [cell for cell, linked in neighbors.items() if len(linked) == 2]
    if len(shafts) != 1:
        return None
    shaft = shafts[0]
    wings = [cell for cell in coords if cell != shaft]
    vecs = [(wr - shaft[0], wc - shaft[1]) for wr, wc in wings]
    if not all(abs(dr) == 1 and abs(dc) == 1 for dr, dc in vecs):
        return None
    sr = vecs[0][0] + vecs[1][0]
    sc = vecs[0][1] + vecs[1][1]
    if (sr == 0) == (sc == 0):
        return None
    if abs(sr) + abs(sc) != 2:
        return None
    direction = (0 if sr == 0 else sr // 2, 0 if sc == 0 else sc // 2)
    return shaft, direction


def _find_target(rectangles, shaft, direction):
    sr, sc = shaft
    dr, dc = direction
    front_r = sr + dr
    front_c = sc + dc
    best = None
    best_dist = None

    for idx, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        if dr == -1:
            if c0 <= front_c <= c1 and r1 <= front_r:
                dist = front_r - r1
            else:
                continue
        elif dr == 1:
            if c0 <= front_c <= c1 and r0 >= front_r:
                dist = r0 - front_r
            else:
                continue
        elif dc == -1:
            if r0 <= front_r <= r1 and c1 <= front_c:
                dist = front_c - c1
            else:
                continue
        else:
            if r0 <= front_r <= r1 and c0 >= front_c:
                dist = c0 - front_c
            else:
                continue

        if best_dist is None or dist < best_dist:
            best = idx
            best_dist = dist
        elif dist == best_dist:
            raise AssertionError("arrow target is ambiguous")

    if best is None:
        raise AssertionError(f"arrow at {shaft} points at no rectangle")
    return best


def _perimeter_coords(bbox):
    r0, c0, r1, c1 = bbox
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint the border of each pointed-at rectangle with the pointing arrow's color, then erase the arrows.

    Input structure:
    - The background is black.
    - Foreground objects are solid monochrome rectangles sized from 2x2 to 4x4.
    - Some rectangles have a nearby 3-pixel monochrome V-arrow: one shaft pixel behind two diagonal wing pixels, pointing up, down, left, or right.

    Rule:
    - Separate the solid rectangles from the 3-pixel V-arrows and decode each arrow's pointing direction from its shaft and wing geometry.
    - From the arrow's empty front-center ray, look straight along that direction and choose the nearest rectangle intersecting that row or column.
    - Recolor that target rectangle's outer perimeter with the arrow color; interior pixels keep their original color, and a 2x2 rectangle changes completely because its whole area is perimeter.
    - Erase every arrow to black. If several arrows point to one rectangle, sort them by the reading order of their shaft pixels and cycle those colors clockwise around the perimeter starting at the rectangle's top-left corner.

    Key insight:
    - The arrows do not transfer their source rectangle's color; they act as directional border-paint commands, and rectangle size determines whether only the border or the whole object changes.
    """
    g = np.asarray(grid, dtype=int)
    rectangles, arrows = _extract_rectangles_and_arrows(g)
    out = g.copy()

    target_to_arrows = defaultdict(list)
    for arrow in arrows:
        target_idx = _find_target(rectangles, arrow["shaft"], arrow["dir"])
        target_to_arrows[target_idx].append(arrow)
        for r, c in arrow["coords"]:
            out[r, c] = BG

    for idx, rect in enumerate(rectangles):
        if idx not in target_to_arrows:
            continue
        perimeter = _perimeter_coords(rect["bbox"])
        ordered = sorted(target_to_arrows[idx], key=lambda arrow: (arrow["shaft"][0], arrow["shaft"][1]))
        colors = [int(arrow["color"]) for arrow in ordered]
        for i, (r, c) in enumerate(perimeter):
            out[r, c] = colors[i % len(colors)]

    return out


def _arrow_cells(shaft, dir_name):
    dr, dc = DIRS[dir_name]
    if dr:
        pr, pc = 0, 1
    else:
        pr, pc = 1, 0
    sr, sc = shaft
    return [
        (sr, sc),
        (sr + dr + pr, sc + dc + pc),
        (sr + dr - pr, sc + dc - pc),
    ]


def _pick_distinct(excluded):
    choices = [color for color in PALETTE if color not in excluded]
    if not choices:
        raise AssertionError("palette exhausted")
    return random.choice(choices)


def _stamp_rect(cells, r0, c0, h, w, color):
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if (r, c) in cells:
                return False
            cells[(r, c)] = color
    return True


def _build_group(target_size, dirs):
    th, tw = target_size
    for _ in range(100):
        cells = {}
        used_colors = set()
        target_color = _pick_distinct(used_colors)
        used_colors.add(target_color)
        if not _stamp_rect(cells, 0, 0, th, tw, target_color):
            continue

        shared_row = None
        shared_col = None
        if set(dirs) == {"left", "right"} and len(dirs) == 2:
            shared_row = random.randint(0, th - 1)
        if set(dirs) == {"up", "down"} and len(dirs) == 2:
            shared_col = random.randint(0, tw - 1)

        ok = True
        for dir_name in dirs:
            sh, sw = random.choice(RECT_DIMS)
            source_color = _pick_distinct(used_colors)
            used_colors.add(source_color)
            arrow_color = _pick_distinct(used_colors)
            used_colors.add(arrow_color)

            if dir_name == "down":
                anchor_c = shared_col if shared_col is not None else random.randint(0, tw - 1)
                c0 = random.randint(anchor_c - sw + 1, anchor_c)
                r0 = -sh - 3
                shaft = (-3, anchor_c)
            elif dir_name == "up":
                anchor_c = shared_col if shared_col is not None else random.randint(0, tw - 1)
                c0 = random.randint(anchor_c - sw + 1, anchor_c)
                r0 = th + 3
                shaft = (th + 2, anchor_c)
            elif dir_name == "right":
                anchor_r = shared_row if shared_row is not None else random.randint(0, th - 1)
                r0 = random.randint(anchor_r - sh + 1, anchor_r)
                c0 = -sw - 3
                shaft = (anchor_r, -3)
            else:
                anchor_r = shared_row if shared_row is not None else random.randint(0, th - 1)
                r0 = random.randint(anchor_r - sh + 1, anchor_r)
                c0 = tw + 3
                shaft = (anchor_r, tw + 2)

            before = dict(cells)
            if not _stamp_rect(cells, r0, c0, sh, sw, source_color):
                ok = False
                break
            for coord in _arrow_cells(shaft, dir_name):
                if coord in cells:
                    ok = False
                    break
                cells[coord] = arrow_color
            if not ok:
                cells = before
                break

        if not ok:
            continue

        coords = list(cells)
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for (r, c), color in cells.items():
            patch[r - r0, c - c0] = color
        return patch, 1 + len(dirs), len(dirs)
    raise RuntimeError("failed to build a group")


def _place_patch(grid, reserved, patch):
    h, w = patch.shape
    gh, gw = grid.shape
    candidates = []
    for r in range(gh - h + 1):
        for c in range(gw - w + 1):
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(gh, r + h + 1)
            cc1 = min(gw, c + w + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r, c))
    if not candidates:
        return False
    r, c = random.choice(candidates)
    mask = patch != BG
    grid[r : r + h, c : c + w][mask] = patch[mask]
    reserved[r : r + h, c : c + w] = True
    return True


def _place_distractor(grid, reserved):
    gh, gw = grid.shape
    h, w = random.choice(RECT_DIMS)
    color = random.choice(PALETTE)
    candidates = []
    for r in range(gh - h + 1):
        for c in range(gw - w + 1):
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(gh, r + h + 1)
            cc1 = min(gw, c + w + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r, c))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    grid[r0 : r0 + h, c0 : c0 + w] = color
    reserved[r0 : r0 + h, c0 : c0 + w] = True
    return True


def _group_specs():
    specs = [
        ((2, 2), [random.choice(tuple(DIRS))]),
        (random.choice(LARGE_DIMS), [random.choice(tuple(DIRS))]),
    ]

    if random.random() < 0.7:
        specs.append((random.choice(RECT_DIMS), list(random.choice((("up", "down"), ("left", "right"))))))
    if random.random() < 0.35:
        dims = random.choice(RECT_DIMS)
        dirs = [random.choice(tuple(DIRS))] if random.random() < 0.6 else list(random.choice((("up", "down"), ("left", "right"))))
        specs.append((dims, dirs))
    random.shuffle(specs)
    return specs


def _gen_attempt():
    specs = _group_specs()
    patches = []
    expected_rects = 0
    expected_arrows = 0
    for dims, dirs in specs:
        patch, rect_count, arrow_count = _build_group(dims, dirs)
        patches.append(patch)
        expected_rects += rect_count
        expected_arrows += arrow_count

    min_h = max(max(patch.shape[0] for patch in patches) + 2, 18)
    min_w = max(max(patch.shape[1] for patch in patches) + 2, 18)
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    random.shuffle(patches)
    for patch in patches:
        if not _place_patch(grid, reserved, patch):
            return None

    distractor_count = random.randint(0, 2)
    for _ in range(distractor_count):
        if _place_distractor(grid, reserved):
            expected_rects += 1

    rectangles, arrows = _extract_rectangles_and_arrows(grid)
    if len(rectangles) != expected_rects or len(arrows) != expected_arrows:
        return None
    for arrow in arrows:
        _find_target(rectangles, arrow["shaft"], arrow["dir"])
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-border puzzle")
