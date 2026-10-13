from __future__ import annotations

from collections import Counter
import random

import numpy as np


OPS = tuple((flip, rot) for flip in (False, True) for rot in range(4))

PAIR_COUNTS = (2, 3, 4)
RECT_HEIGHT_RANGE = (4, 7)
RECT_WIDTH_RANGE = (4, 8)
GRID_HEIGHT_RANGE = (23, 30)
GRID_WIDTH_RANGE = (23, 30)
GEN_RETRIES = 400
PLACE_RETRIES = 300
PROFILE_RETRIES = 80
COLOR_POOL = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _transform(arr: np.ndarray, flip_lr: bool, rot_k: int) -> np.ndarray:
    out = np.fliplr(arr) if flip_lr else arr
    return np.rot90(out, rot_k)


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 6

    bg = _get_bg(g)
    assert int(g[0, 0]) == bg

    top_markers: dict[int, int] = {}
    left_markers: dict[int, int] = {}
    marker_mask = np.zeros((h, w), dtype=bool)

    for c in range(1, w):
        color = int(g[0, c])
        if color == bg:
            continue
        assert color not in top_markers and color not in left_markers
        top_markers[color] = c
        marker_mask[0, c] = True

    for r in range(1, h):
        color = int(g[r, 0])
        if color == bg:
            continue
        assert color not in top_markers and color not in left_markers
        left_markers[color] = r
        marker_mask[r, 0] = True

    assert top_markers and left_markers
    assert set(top_markers).isdisjoint(left_markers)
    assert len(top_markers) == len(left_markers)

    interior = g.copy()
    interior[marker_mask] = bg
    fragment_colors = {int(v) for v in np.unique(interior) if int(v) != bg}
    assert fragment_colors == set(top_markers) | set(left_markers)

    pieces: dict[int, np.ndarray] = {}
    for color in fragment_colors:
        mask = interior == color
        comps = _components(mask)
        assert len(comps) == 1
        pts = comps[0]
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        piece = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert piece.any()
        pieces[color] = piece

    return {
        "grid": g,
        "bg": bg,
        "shape": g.shape,
        "top_markers": top_markers,
        "left_markers": left_markers,
        "pieces": pieces,
    }


def _merge_option(top_mask: np.ndarray, left_mask: np.ndarray, top_color: int, left_color: int) -> np.ndarray | None:
    h, w = top_mask.shape
    hl, wl = left_mask.shape
    if hl != h - 1 or wl > w:
        return None
    if not np.all(top_mask[0]):
        return None
    if not np.all(left_mask[:, 0]):
        return None

    patch = np.full((h, w), -1, dtype=int)
    patch[top_mask] = top_color

    for r in range(hl):
        row_left = np.zeros(w, dtype=bool)
        row_left[:wl] = left_mask[r]
        row_top = top_mask[r + 1]
        if np.any(row_top & row_left):
            return None
        if not np.all(row_top | row_left):
            return None
        patch[r + 1, :wl][left_mask[r]] = left_color

    if np.any(patch == -1):
        return None
    return patch


def _pair_options(scene: dict, top_color: int, left_color: int) -> list[dict]:
    top_piece = scene["pieces"][top_color]
    left_piece = scene["pieces"][left_color]
    options: list[dict] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for top_op in OPS:
        top_mask = _transform(top_piece, *top_op)
        for left_op in OPS:
            left_mask = _transform(left_piece, *left_op)
            patch = _merge_option(top_mask, left_mask, top_color, left_color)
            if patch is None:
                continue
            sig = (patch.shape, patch.tobytes())
            if sig in seen:
                continue
            seen.add(sig)
            options.append(
                {
                    "top_color": top_color,
                    "left_color": left_color,
                    "patch": patch,
                }
            )
    return options


def _solve_scene(scene: dict) -> np.ndarray:
    top_colors = list(scene["top_markers"])
    left_colors = list(scene["left_markers"])

    candidates: dict[tuple[int, int], list[dict]] = {}
    for top_color in top_colors:
        for left_color in left_colors:
            candidates[(top_color, left_color)] = _pair_options(scene, top_color, left_color)

    for top_color in top_colors:
        assert any(candidates[(top_color, left_color)] for left_color in left_colors)
    for left_color in left_colors:
        assert any(candidates[(top_color, left_color)] for top_color in top_colors)

    order = sorted(
        top_colors,
        key=lambda color: sum(bool(candidates[(color, left_color)]) for left_color in left_colors),
    )

    solutions: list[dict[int, dict]] = []
    used_left: set[int] = set()
    current: dict[int, dict] = {}

    def dfs(idx: int) -> None:
        if len(solutions) > 1:
            return
        if idx == len(order):
            solutions.append(current.copy())
            return

        top_color = order[idx]
        for left_color in left_colors:
            if left_color in used_left:
                continue
            for option in candidates[(top_color, left_color)]:
                used_left.add(left_color)
                current[top_color] = option
                dfs(idx + 1)
                del current[top_color]
                used_left.remove(left_color)
                if len(solutions) > 1:
                    return

    dfs(0)
    assert len(solutions) == 1
    assignment = solutions[0]

    out = np.full(scene["shape"], scene["bg"], dtype=int)
    for top_color, option in assignment.items():
        patch = option["patch"]
        left_color = option["left_color"]
        row = scene["left_markers"][left_color]
        col = scene["top_markers"][top_color]
        h, w = patch.shape
        assert row + h <= out.shape[0] and col + w <= out.shape[1]
        region = out[row : row + h, col : col + w]
        assert np.all(region == scene["bg"])
        region[:] = patch

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered colored fragments by complementary silhouette, merge each pair into a filled rectangle,
    and place the merged rectangle at the intersection of its left-edge and top-edge markers.

    Input structure
    - Background is the most common color.
    - Non-background interior cells are monochrome connected fragments, one fragment per color.
    - Every fragment color appears exactly once as a single-pixel marker, either on the top edge or on the left edge.
    - A valid pair always consists of one top-edge color and one left-edge color.

    Rule
    - Match fragments whose shapes can be rotated/flipped and then packed together into one solid rectangle.
    - Keep each pixel's original fragment color in the merged rectangle.
    - Place the merged rectangle so its top-left corner is at (left-marker row, top-marker column).
    - Output only the merged rectangles on the background.
    """
    scene = _extract_scene(grid)
    return _solve_scene(scene)


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 + margin < br0 or br1 + margin < ar0:
        return True
    if ac1 + margin < bc0 or bc1 + margin < ac0:
        return True
    return False


def _build_pair(top_color: int, left_color: int, rng: random.Random) -> dict | None:
    for _ in range(PROFILE_RETRIES):
        h = rng.randint(*RECT_HEIGHT_RANGE)
        w = rng.randint(*RECT_WIDTH_RANGE)
        profile = [0] + [rng.randint(1, w - 1) for _ in range(h - 1)]
        if len(set(profile[1:])) == 1 and rng.random() < 0.75:
            row = rng.randrange(1, h)
            delta = rng.choice((-1, 1))
            profile[row] = max(1, min(w - 1, profile[row] + delta))

        left_area = sum(profile[1:])
        total_area = h * w
        top_area = total_area - left_area
        if min(top_area, left_area) < max(4, total_area // 5):
            continue

        top_mask = np.zeros((h, w), dtype=bool)
        top_mask[0, :] = True
        for r in range(1, h):
            top_mask[r, profile[r] :] = True

        left_width = max(profile[1:])
        left_mask = np.zeros((h - 1, left_width), dtype=bool)
        for r in range(h - 1):
            left_mask[r, : profile[r + 1]] = True

        assert len(_components(top_mask)) == 1
        assert len(_components(left_mask)) == 1

        patch = np.full((h, w), top_color, dtype=int)
        for r in range(1, h):
            patch[r, : profile[r]] = left_color

        return {
            "top_color": top_color,
            "left_color": left_color,
            "top_mask": top_mask,
            "left_mask": left_mask,
            "patch": patch,
        }
    return None


def _place_targets(pair_specs: list[dict], grid_h: int, grid_w: int, rng: random.Random) -> bool:
    placed_boxes: list[tuple[int, int, int, int]] = []
    used_rows: list[int] = []
    used_cols: list[int] = []
    order = sorted(range(len(pair_specs)), key=lambda idx: pair_specs[idx]["patch"].size, reverse=True)

    for idx in order:
        patch = pair_specs[idx]["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(PLACE_RETRIES):
            row = rng.randint(1, grid_h - h)
            col = rng.randint(1, grid_w - w)
            box = (row, col, row + h - 1, col + w - 1)
            if any(abs(row - prev) <= 1 for prev in used_rows):
                continue
            if any(abs(col - prev) <= 1 for prev in used_cols):
                continue
            if any(not _boxes_separate(box, other, margin=1) for other in placed_boxes):
                continue
            pair_specs[idx]["target"] = (row, col)
            placed_boxes.append(box)
            used_rows.append(row)
            used_cols.append(col)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_fragments(
    grid: np.ndarray,
    blocked: np.ndarray,
    pair_specs: list[dict],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    items = []
    for spec in pair_specs:
        items.append(
            {
                "color": spec["top_color"],
                "mask": _transform(spec["top_mask"], *rng.choice(OPS)),
            }
        )
        items.append(
            {
                "color": spec["left_color"],
                "mask": _transform(spec["left_mask"], *rng.choice(OPS)),
            }
        )

    items.sort(key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []
    occupied = np.zeros((h, w), dtype=bool)

    for item in items:
        mask = item["mask"]
        ph, pw = mask.shape
        coords = np.argwhere(mask)
        placed = False
        for _ in range(PLACE_RETRIES):
            row = rng.randint(1, h - ph)
            col = rng.randint(1, w - pw)
            box = (row, col, row + ph - 1, col + pw - 1)
            if any(not _boxes_separate(box, other, margin=1) for other in placed_boxes):
                continue
            rr = row + coords[:, 0]
            cc = col + coords[:, 1]
            if np.any(blocked[rr, cc]) or np.any(occupied[rr, cc]):
                continue
            grid[rr, cc] = item["color"]
            occupied[rr, cc] = True
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_output(pair_specs: list[dict], bg: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in pair_specs:
        row, col = spec["target"]
        patch = spec["patch"]
        h, w = patch.shape
        assert np.all(out[row : row + h, col : col + w] == bg)
        out[row : row + h, col : col + w] = patch
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(COLOR_POOL)
    colors = [color for color in COLOR_POOL if color != bg]
    n_pairs = rng.choice(PAIR_COUNTS)
    chosen = rng.sample(colors, 2 * n_pairs)
    rng.shuffle(chosen)

    pair_specs: list[dict] = []
    for idx in range(n_pairs):
        a = chosen[2 * idx]
        b = chosen[2 * idx + 1]
        if rng.random() < 0.5:
            top_color, left_color = a, b
        else:
            top_color, left_color = b, a
        spec = _build_pair(top_color, left_color, rng)
        if spec is None:
            return None
        pair_specs.append(spec)

    min_h = max(max(spec["patch"].shape[0] for spec in pair_specs) + 3, GRID_HEIGHT_RANGE[0])
    min_w = max(max(spec["patch"].shape[1] for spec in pair_specs) + 3, GRID_WIDTH_RANGE[0])
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    grid_h = rng.randint(min_h, GRID_HEIGHT_RANGE[1])
    grid_w = rng.randint(min_w, GRID_WIDTH_RANGE[1])
    if not _place_targets(pair_specs, grid_h, grid_w, rng):
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in pair_specs:
        row, col = spec["target"]
        patch_h, patch_w = spec["patch"].shape
        blocked[row : row + patch_h, col : col + patch_w] = True
        if 1 < grid_w:
            blocked[row, 1] = True
        if 1 < grid_h:
            blocked[1, col] = True

    if not _place_fragments(grid, blocked, pair_specs, rng):
        return None

    for spec in pair_specs:
        row, col = spec["target"]
        assert grid[0, col] == bg
        assert grid[row, 0] == bg
        grid[0, col] = spec["top_color"]
        grid[row, 0] = spec["left_color"]

    expected = _expected_output(pair_specs, bg, grid.shape)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate ARC-style inputs for the fragment-pairing rectangle puzzle.
    """
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
