import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp = np.zeros((h, w), dtype=bool)
            for rr, cc in cells:
                comp[rr, cc] = True
            comps.append(comp)
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _crop(mask: np.ndarray) -> np.ndarray:
    r0, r1, c0, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _extract_objects(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    h, w = g.shape

    candidates: list[int] = []
    for c in range(w):
        if int(g[0, c]) == bg:
            continue
        if not np.all(g[:, c] == g[0, c]):
            continue
        left = g[:, :c] != bg
        right = g[:, c + 1 :] != bg
        if left.any() and right.any() and len(_components(left)) == 1 and len(_components(right)) == 1:
            candidates.append(c)
    assert len(candidates) == 1

    sep = candidates[0]
    left_mask = g[:, :sep] != bg
    right_mask = g[:, sep + 1 :] != bg
    left_comp = _components(left_mask)[0]
    right_comp = _components(right_mask)[0]
    return bg, sep, left_comp, right_comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the left-panel object by the area ratio encoded by the right panel.

    Input structure:
    - A dominant background contains one full-height separator column.
    - One 4-connected foreground object lies in each panel, with left area <= right area.
    - The area ratio right/left is an integer K.

    Rule:
    - Compute K = area(right) / area(left).
    - Scale the left object by K in both axes by expanding each pixel into a KxK block.
    - Output only the scaled left object, tightly cropped to its scaled bbox.
    """
    bg, sep, left_comp, right_comp = _extract_objects(grid)
    left_area = int(left_comp.sum())
    right_area = int(right_comp.sum())
    assert left_area > 0 and right_area > 0
    assert right_area % left_area == 0
    scale = right_area // left_area
    assert scale >= 1

    g = np.asarray(grid, dtype=int)
    left_color = int(g[:, :sep][left_comp][0])
    left_crop = _crop(left_comp)
    scaled_mask = np.kron(left_crop, np.ones((scale, scale), dtype=bool))
    out = np.full(scaled_mask.shape, bg, dtype=int)
    out[scaled_mask] = left_color
    return out


GENERATION_ATTEMPTS = 400
SCALE_CHOICES = (1, 2, 3)
AREA_RANGE = (2, 7)
MARGIN_RANGE = (1, 3)
SLACK_RANGE = (0, 4)


def _grow_connected_mask(rng: np.random.Generator, target_area: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(120):
        box_h = int(rng.integers(1, max_h + 1))
        box_w = int(rng.integers(1, max_w + 1))
        if box_h * box_w < target_area:
            continue
        if box_h * box_w > max(target_area + 8, target_area * 3):
            continue

        start = (int(rng.integers(box_h)), int(rng.integers(box_w)))
        cells = {start}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)
        if len(cells) != target_area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        cropped = _crop(mask)
        if cropped.shape[0] > max_h or cropped.shape[1] > max_w:
            continue
        return cropped
    return None


def _sample_mask(rng: np.random.Generator, target_area: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(120):
        mask = _grow_connected_mask(rng, target_area, max_h, max_w)
        if mask is None:
            continue
        if mask.shape[0] == 1 and mask.shape[1] == 1 and target_area > 1:
            continue
        return mask
    return None


def _place_mask(canvas: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    region = canvas[top : top + h, left : left + w]
    region[mask] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    scale = int(rng.choice(SCALE_CHOICES))
    left_area = int(rng.integers(AREA_RANGE[0], AREA_RANGE[1] + 1))
    right_area = left_area * scale

    left_mask = _sample_mask(rng, left_area, max_h=5, max_w=5)
    right_mask = _sample_mask(rng, right_area, max_h=8, max_w=8)
    if left_mask is None or right_mask is None:
        return None

    left_h, left_w = left_mask.shape
    right_h, right_w = right_mask.shape

    top_pad_left = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    bot_pad_left = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    left_pad_left = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    right_pad_left = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))

    top_pad_right = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    bot_pad_right = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    left_pad_right = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    right_pad_right = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))

    height = max(
        left_h + top_pad_left + bot_pad_left,
        right_h + top_pad_right + bot_pad_right,
    ) + int(rng.integers(SLACK_RANGE[0], SLACK_RANGE[1] + 1))

    left_panel_w = left_w + left_pad_left + right_pad_left + int(rng.integers(SLACK_RANGE[0], SLACK_RANGE[1] + 1))
    right_panel_w = right_w + left_pad_right + right_pad_right + int(rng.integers(SLACK_RANGE[0], SLACK_RANGE[1] + 1))
    width = left_panel_w + 1 + right_panel_w

    palette = list(range(10))
    bg, sep_color, left_color, right_color = rng.choice(palette, size=4, replace=False).tolist()
    grid = np.full((height, width), bg, dtype=int)

    sep = left_panel_w
    grid[:, sep] = sep_color

    top_left = int(rng.integers(1, height - left_h))
    col_left = int(rng.integers(1, left_panel_w - left_w))
    top_right = int(rng.integers(1, height - right_h))
    col_right = sep + 1 + int(rng.integers(1, right_panel_w - right_w))

    _place_mask(grid, left_mask, left_color, top_left, col_left)
    _place_mask(grid, right_mask, right_color, top_right, col_right)

    try:
        out = main(grid)
    except Exception:
        return None

    expected_shape = (left_h * scale, left_w * scale)
    if out.shape != expected_shape:
        return None
    if int((grid != bg).sum()) <= height:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
