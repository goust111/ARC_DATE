import numpy as np


BLACK = 0
MAX_SIZE = 30


def _neighbors(r, c, h, w, diagonals=False):
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonals:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bbox_from_mask(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return None
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_full_box(mask):
    return (
        mask.any()
        and (~mask).any()
        and mask[0].any()
        and mask[-1].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _is_connected(mask, diagonals=False):
    h, w = mask.shape
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(int(x) for x in points[0])
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, h, w, diagonals=diagonals):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _component_items(grid):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    items = []
    for r in range(h):
        for c in range(w):
            color = grid[r, c]
            if color == BLACK or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for cr, cc in cells:
                mask[cr - r0, cc - c0] = True
            items.append({"color": color, "pos": (r0, c0), "mask": mask})
    return items


def _color_group_items(grid):
    items = []
    for color in sorted(int(x) for x in np.unique(grid) if x != BLACK):
        mask_all = grid == color
        box = _bbox_from_mask(mask_all)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        items.append(
            {
                "color": color,
                "pos": (r0, c0),
                "mask": mask_all[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )
    return items


def _assemble_from_items(items, dtype):
    if len(items) == 0 or len(items) % 2:
        return None

    first_shape = items[0]["mask"].shape
    if any(item["mask"].shape != first_shape for item in items):
        return None

    height, width = first_shape
    unused = set(range(len(items)))
    pairs = []

    order = sorted(range(len(items)), key=lambda i: items[i]["pos"])
    for i in order:
        if i not in unused:
            continue
        complement = ~items[i]["mask"]
        candidates = [
            j
            for j in unused
            if j != i and np.array_equal(items[j]["mask"], complement)
        ]
        if not candidates:
            return None
        j = min(candidates, key=lambda idx: items[idx]["pos"])
        unused.remove(i)
        unused.remove(j)
        pairs.append((i, j, min(items[i]["pos"], items[j]["pos"])))

    pairs.sort(key=lambda pair: pair[2])
    out_width = len(pairs) * width + max(0, len(pairs) - 1)
    out = np.zeros((height, out_width), dtype=dtype)
    for k, (i, j, _) in enumerate(pairs):
        c0 = k * (width + 1)
        rect = out[:, c0 : c0 + width]
        rect[items[i]["mask"]] = items[i]["color"]
        rect[items[j]["mask"]] = items[j]["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary colored fragments and rebuild each pair as a filled rectangle.

    Input structure:
    The grid has a black background and an even number of colored fragments. Each
    fragment has one visible color; fragments share a common bounding-box size and
    are scattered without overlap.

    Rule:
    Extract each fragment mask in its tight bounding box, pair masks that are exact
    complements within the shared box, and draw one full rectangle for each pair.
    Cells belonging to either fragment keep that fragment's color. The completed
    rectangles are placed left to right with one black column between them, ordered
    by the top-left position of the earlier fragment in each pair.

    Key insight:
    The pairing is geometric: matching colors is insufficient, because the two
    fragments must have complementary shapes that tile the same rectangle.
    """
    strict = _assemble_from_items(_component_items(grid), grid.dtype)
    if strict is not None:
        return strict

    grouped = _assemble_from_items(_color_group_items(grid), grid.dtype)
    if grouped is not None:
        return grouped

    return np.zeros((1, 1), dtype=grid.dtype)


def _mask_signature(mask):
    return mask.astype(np.uint8).tobytes()


def _valid_generated_mask(mask):
    comp = ~mask
    min_cells = max(3, mask.size // 5)
    if int(mask.sum()) < min_cells or int(comp.sum()) < min_cells:
        return False
    return (
        _touches_full_box(mask)
        and _touches_full_box(comp)
        and _is_connected(mask, diagonals=True)
        and _is_connected(comp, diagonals=True)
    )


def _random_piece_mask(h, w, used, rng):
    for _ in range(12000):
        density = float(rng.uniform(0.35, 0.65))
        mask = rng.random((h, w)) < density
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if h == w:
            mask = np.rot90(mask, int(rng.integers(0, 4)))
        comp = ~mask
        sig = _mask_signature(mask)
        comp_sig = _mask_signature(comp)
        if sig in used or comp_sig in used:
            continue
        if _valid_generated_mask(mask):
            used.add(sig)
            used.add(comp_sig)
            return mask.copy()

    rr, cc = np.indices((h, w))
    for a in range(1, 5):
        for b in range(1, 5):
            for modulus in range(2, 6):
                for threshold in range(1, modulus):
                    for offset in range(modulus):
                        mask = ((a * rr + b * cc + offset) % modulus) < threshold
                        comp = ~mask
                        sig = _mask_signature(mask)
                        comp_sig = _mask_signature(comp)
                        if sig in used or comp_sig in used:
                            continue
                        if _valid_generated_mask(mask):
                            used.add(sig)
                            used.add(comp_sig)
                            return mask.copy()

    raise RuntimeError("could not generate a complementary mask pair")


def _slot_layouts(n_items, h, w):
    layouts = []
    for rows in range(1, n_items + 1):
        for cols in range(1, n_items + 1):
            if rows * cols < n_items:
                continue
            base_h = rows * h + rows - 1
            base_w = cols * w + cols - 1
            if base_h <= MAX_SIZE and base_w <= MAX_SIZE:
                layouts.append((rows, cols, base_h, base_w))
    return layouts


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(100):
        box_h = int(rng.integers(4, 8))
        box_w = int(rng.integers(4, 8))
        max_pairs = min(4, (MAX_SIZE + 1) // (box_w + 1))
        if max_pairs < 2:
            continue
        pair_count = int(rng.integers(2, max_pairs + 1))
        layouts = _slot_layouts(pair_count * 2, box_h, box_w)
        if layouts:
            break
    else:
        box_h, box_w, pair_count = 5, 5, 2
        layouts = _slot_layouts(pair_count * 2, box_h, box_w)

    rows, cols, base_h, base_w = layouts[int(rng.integers(0, len(layouts)))]
    extra_h = int(rng.integers(0, MAX_SIZE - base_h + 1))
    extra_w = int(rng.integers(0, MAX_SIZE - base_w + 1))
    grid_h = base_h + extra_h
    grid_w = base_w + extra_w
    top_margin = int(rng.integers(0, extra_h + 1))
    left_margin = int(rng.integers(0, extra_w + 1))

    slots = []
    for sr in range(rows):
        for sc in range(cols):
            slots.append((top_margin + sr * (box_h + 1), left_margin + sc * (box_w + 1)))
    rng.shuffle(slots)

    colors = np.arange(1, 10)
    rng.shuffle(colors)
    used_masks = set()
    pieces = []
    for pair_idx in range(pair_count):
        mask = _random_piece_mask(box_h, box_w, used_masks, rng)
        if rng.random() < 0.5:
            mask = ~mask
        pieces.append((mask, int(colors[2 * pair_idx])))
        pieces.append((~mask, int(colors[2 * pair_idx + 1])))

    rng.shuffle(pieces)
    grid = np.zeros((grid_h, grid_w), dtype=np.int64)
    for (mask, color), (r, c) in zip(pieces, slots):
        window = grid[r : r + box_h, c : c + box_w]
        window[mask] = color
    return grid
