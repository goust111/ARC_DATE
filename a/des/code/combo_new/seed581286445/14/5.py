from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
BRANCHES = ("horizontal", "vertical", "diagonal")
WIDE_DIMS = [(h, w) for h in range(2, 6) for w in range(2, 6) if w > h]
TALL_DIMS = [(h, w) for h in range(2, 6) for w in range(2, 6) if h > w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the large frame with an integer-scaled copy of the outside template selected by the marker-line direction.

    Input structure:
    - The background is black.
    - One large grey one-pixel-thick hollow rectangle appears somewhere in the grid.
    - Exactly two same-colored single-pixel markers lie inside that frame.
    - Outside the frame there are small non-grey templates, including exactly one wider-than-tall template and exactly one taller-than-wide template.

    Rule:
    - If the two markers share a row, choose the horizontal template.
    - If the two markers share a column, choose the vertical template.
    - Otherwise choose the template whose bounding box has the larger area.
    - Crop the chosen template to its bounding box, scale every cell into an n x n block so that the scaled template bbox matches the frame interior exactly, and replace the whole frame interior with that scaled pattern.
    - Leave the frame border and the unchosen outside template unchanged.

    Key insight:
    - The marker pair encodes which template to use, while the frame interior dimensions reveal the unique integer scale factor for that template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _corner_mode(g)
    frame = _find_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame["bbox"]
    frame_color = frame["color"]

    interior = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    marker_coords = np.argwhere((interior != bg) & (interior != frame_color))
    assert len(marker_coords) == 2, "expected exactly two frame markers"
    marker_colors = {int(interior[r, c]) for r, c in marker_coords}
    assert len(marker_colors) == 1, "markers must share one color"

    wide_template, tall_template = _extract_templates(g, bg, frame["bbox"], frame_color)

    (mr0, mc0), (mr1, mc1) = [(int(r), int(c)) for r, c in marker_coords]
    if mr0 == mr1:
        chosen = wide_template
    elif mc0 == mc1:
        chosen = tall_template
    else:
        assert wide_template["area"] != tall_template["area"], "diagonal case needs unequal template areas"
        chosen = wide_template if wide_template["area"] > tall_template["area"] else tall_template

    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1
    tmpl_h, tmpl_w = chosen["patch"].shape
    assert inner_h % tmpl_h == 0 and inner_w % tmpl_w == 0, "frame interior must be an integer template scale"
    scale_h = inner_h // tmpl_h
    scale_w = inner_w // tmpl_w
    assert scale_h == scale_w and scale_h >= 1

    scaled = np.kron(chosen["patch"], np.ones((scale_h, scale_h), dtype=int))
    out = g.copy()
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = scaled
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    branch = rng.choice(BRANCHES)
    larger_kind = rng.choice(("wide", "tall"))

    dims = _sample_template_dims(rng, larger_kind)
    if dims is None:
        return None
    wide_dims, tall_dims = dims

    wide_mask = _random_connected_mask(*wide_dims, rng)
    tall_mask = _random_connected_mask(*tall_dims, rng)
    if wide_mask is None or tall_mask is None:
        return None

    wide_color, tall_color, marker_color = rng.sample(COLOR_POOL, 3)
    wide_patch = np.where(wide_mask, wide_color, BG).astype(int)
    tall_patch = np.where(tall_mask, tall_color, BG).astype(int)

    if branch == "horizontal":
        chosen_kind = "wide"
    elif branch == "vertical":
        chosen_kind = "tall"
    else:
        chosen_kind = larger_kind

    chosen_patch = wide_patch if chosen_kind == "wide" else tall_patch
    chosen_h, chosen_w = chosen_patch.shape
    n_max_h = (24 - wide_patch.shape[0] - tall_patch.shape[0]) // chosen_h
    n_max_w = 26 // chosen_w
    n_max = min(n_max_h, n_max_w)
    if n_max < 2:
        return None
    scale = rng.randint(2, n_max)

    inner_h = chosen_h * scale
    inner_w = chosen_w * scale
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    if rng.random() < 0.5:
        upper_name, upper_patch = "wide", wide_patch
        lower_name, lower_patch = "tall", tall_patch
    else:
        upper_name, upper_patch = "tall", tall_patch
        lower_name, lower_patch = "wide", wide_patch
    assert upper_name != lower_name

    min_h = upper_patch.shape[0] + lower_patch.shape[0] + frame_h + 4
    min_w = max(frame_w, upper_patch.shape[1], lower_patch.shape[1]) + 2
    if min_h > 30 or min_w > 30:
        return None

    extra_h = rng.randint(0, min(6, 30 - min_h))
    extra_w = rng.randint(0, min(8, 30 - min_w))
    top_extra, gap1_extra, gap2_extra, bottom_extra = _split_extra(extra_h, 4, rng)

    top_margin = 1 + top_extra
    gap1 = 1 + gap1_extra
    gap2 = 1 + gap2_extra
    bottom_margin = 1 + bottom_extra

    upper_r = top_margin
    frame_r = upper_r + upper_patch.shape[0] + gap1
    lower_r = frame_r + frame_h + gap2

    grid_h = lower_r + lower_patch.shape[0] + bottom_margin
    grid_w = min_w + extra_w
    assert grid_h <= 30 and grid_w <= 30

    frame_c = rng.randint(1, grid_w - frame_w - 1)
    upper_c = rng.randint(1, grid_w - upper_patch.shape[1] - 1)
    lower_c = rng.randint(1, grid_w - lower_patch.shape[1] - 1)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _paste_patch(grid, upper_patch, upper_r, upper_c)
    _paste_patch(grid, lower_patch, lower_r, lower_c)
    _draw_frame(grid, frame_r, frame_c, frame_h, frame_w, FRAME_COLOR)

    markers = _sample_markers(inner_h, inner_w, branch, rng)
    if markers is None:
        return None
    (r0, c0), (r1, c1) = markers
    grid[frame_r + 1 + r0, frame_c + 1 + c0] = marker_color
    grid[frame_r + 1 + r1, frame_c + 1 + c1] = marker_color

    expected = grid.copy()
    expected[frame_r + 1 : frame_r + 1 + inner_h, frame_c + 1 : frame_c + 1 + inner_w] = _scale_patch(chosen_patch, scale)

    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def _corner_mode(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_frame(grid: np.ndarray, bg: int) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue

            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True

            if len(coords) != int(border.sum()):
                continue
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[~border] == color):
                continue
            candidates.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "area": h * w,
                }
            )

    assert candidates, "expected a hollow rectangle frame"
    candidates.sort(key=lambda item: int(item["area"]))
    return candidates[-1]


def _extract_templates(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    frame_color: int,
) -> tuple[dict[str, object], dict[str, object]]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    templates: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, frame_color):
            continue
        for coords in _components((grid == color) & outside):
            br0, bc0, br1, bc1 = _bbox(coords)
            patch = grid[br0 : br1 + 1, bc0 : bc1 + 1].copy()
            h = br1 - br0 + 1
            w = bc1 - bc0 + 1
            templates.append(
                {
                    "color": color,
                    "bbox": (br0, bc0, br1, bc1),
                    "patch": patch,
                    "height": h,
                    "width": w,
                    "area": h * w,
                }
            )

    wide = [item for item in templates if int(item["width"]) > int(item["height"])]
    tall = [item for item in templates if int(item["height"]) > int(item["width"])]
    assert len(wide) == 1 and len(tall) == 1, "expected one horizontal and one vertical template"
    return wide[0], tall[0]


def _sample_template_dims(
    rng: random.Random,
    larger_kind: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(200):
        wide_dims = rng.choice(WIDE_DIMS)
        tall_dims = rng.choice(TALL_DIMS)
        wide_area = wide_dims[0] * wide_dims[1]
        tall_area = tall_dims[0] * tall_dims[1]
        if wide_area == tall_area:
            continue
        if larger_kind == "wide" and wide_area <= tall_area:
            continue
        if larger_kind == "tall" and tall_area <= wide_area:
            continue
        return wide_dims, tall_dims
    return None


def _random_connected_mask(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    total = h * w
    min_cells = max(h + w - 1, (total + 1) // 2)
    for allow_full in (False, True):
        max_cells = total if allow_full or total <= 4 else total - 1
        if min_cells > max_cells:
            continue
        for _ in range(300):
            target = rng.randint(min_cells, max_cells)
            start = (rng.randrange(h), rng.randrange(w))
            cells = {start}
            while len(cells) < target:
                frontier = []
                frontier_set = set()
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c):
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in frontier_set:
                            frontier.append((nr, nc))
                            frontier_set.add((nr, nc))
                if not frontier:
                    break
                cells.add(rng.choice(frontier))
            if len(cells) != target:
                continue

            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
                continue
            if not allow_full and total > 4 and mask.all():
                continue
            return mask
    return None


def _split_extra(extra: int, parts: int, rng: random.Random) -> list[int]:
    out = [0] * parts
    for _ in range(extra):
        out[rng.randrange(parts)] += 1
    return out


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    target[mask] = patch[mask]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _sample_markers(
    inner_h: int,
    inner_w: int,
    branch: str,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if inner_h < 2 or inner_w < 2:
        return None
    if branch == "horizontal":
        row = rng.randrange(inner_h)
        c0, c1 = sorted(rng.sample(range(inner_w), 2))
        return (row, c0), (row, c1)
    if branch == "vertical":
        col = rng.randrange(inner_w)
        r0, r1 = sorted(rng.sample(range(inner_h), 2))
        return (r0, col), (r1, col)

    for _ in range(100):
        r0, c0 = rng.randrange(inner_h), rng.randrange(inner_w)
        r1, c1 = rng.randrange(inner_h), rng.randrange(inner_w)
        if (r0, c0) != (r1, c1) and r0 != r1 and c0 != c1:
            return (r0, c0), (r1, c1)
    return None


def _scale_patch(patch: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(patch, np.ones((scale, scale), dtype=int))
