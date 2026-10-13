import random
from collections import defaultdict

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        pts = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_separator(g: np.ndarray) -> int:
    h, w = g.shape
    gray_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]
    if len(gray_cols) == 1:
        return int(gray_cols[0])

    solid_cols = []
    for c in range(w):
        col = g[:, c]
        if np.all(col == col[0]) and int(col[0]) != BG:
            solid_cols.append(c)
    assert len(solid_cols) == 1, "expected one full-height separator column"
    return int(solid_cols[0])


def _find_containers(g: np.ndarray, sep_col: int) -> dict[int, tuple[int, int, int, int]]:
    left = g[:, :sep_col]
    colors = set(map(int, left.ravel())) - {BG, SEP}
    containers = {}
    for color in colors:
        coords = np.argwhere(left == color)
        assert coords.size, "container color must have pixels"
        comps = _components(left == color)
        assert len(comps) == 1, "each container color must form one rectangle border"

        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "container must have an interior"

        sub = left[r0 : r1 + 1, c0 : c1 + 1]
        perim = np.zeros_like(sub, dtype=bool)
        perim[0, :] = True
        perim[-1, :] = True
        perim[:, 0] = True
        perim[:, -1] = True
        assert np.all(sub[perim] == color), "container border must be solid"
        assert np.all(sub[~perim] == BG), "container interior must be empty"
        assert int(coords.shape[0]) == int(perim.sum()), "container color cannot have stray pixels"
        containers[int(color)] = (r0, c0, r1, c1)
    assert containers, "expected at least one container"
    return containers


def _extract_fragments(g: np.ndarray, sep_col: int, containers: dict[int, tuple[int, int, int, int]]):
    right = g[:, sep_col + 1 :]
    by_color = defaultdict(list)
    colors = set(map(int, right.ravel())) - {BG, SEP}
    for color in colors:
        assert color in containers, "fragment color must match a container color"
        for coords in _components(right == color):
            assert 2 <= int(coords.shape[0]) <= 4, "fragments must be small connected blocks"
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            by_color[int(color)].append(
                {
                    "top": int(r0),
                    "left": int(c0 + sep_col + 1),
                    "mask": mask,
                }
            )
    return by_color


def main(grid: np.ndarray) -> np.ndarray:
    """Sort colored fragments into matching framed containers and stack them by source height.

    Input structure:
    - The background is black. One full-height gray column separates a left container area from a right fragment area.
    - The left area contains non-overlapping one-pixel hollow rectangular frames, each in a distinct color and with black interiors.
    - The right area contains small monochrome 4-connected fragments; every fragment color matches exactly one frame color.

    Rule:
    - Clear the entire right area back to the black background while preserving the gray separator and the colored frame borders.
    - Match fragments to containers by shared color.
    - For each color, place the fragments inside the matching frame from bottom to top, using the original top row in the right area as the order: lower source fragments settle lower.
    - Each fragment keeps its cropped shape, is horizontally centered in the frame interior, and uses the left-biased center when the parity is uneven.

    Key insight:
    - Color gives the destination container, but the vertical source coordinate gives the stack order, so the right-side fragments visually settle into their same-colored frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_col = _find_separator(g)
    containers = _find_containers(g, sep_col)
    fragments = _extract_fragments(g, sep_col, containers)

    out = g.copy()
    out[:, sep_col + 1 :] = BG

    for color, (r0, c0, r1, c1) in containers.items():
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        group = fragments.get(color, [])
        tops = [frag["top"] for frag in group]
        assert len(tops) == len(set(tops)), "same-colored fragments need unique top rows"

        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        cursor = r1 - 1
        for frag in sorted(group, key=lambda item: (-item["top"], item["left"])):
            mask = frag["mask"]
            mh, mw = mask.shape
            assert mh <= interior_h and mw <= interior_w, "fragment must fit in its container"
            rr0 = cursor - mh + 1
            cc0 = c0 + 1 + (interior_w - mw) // 2
            assert rr0 >= r0 + 1, "stack must fit inside the container"
            patch = out[rr0 : rr0 + mh, cc0 : cc0 + mw]
            assert np.all((patch[mask] == BG) | (patch[mask] == color))
            patch[mask] = color
            cursor = rr0 - 1

    return out


SHAPE_TEMPLATES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
]


def _normalize_cells(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _random_shape(rng: random.Random) -> np.ndarray:
    cells = list(rng.choice(SHAPE_TEMPLATES))
    if rng.random() < 0.5:
        max_c = max(c for _r, c in cells)
        cells = [(r, max_c - c) for r, c in cells]
    if rng.random() < 0.5:
        max_r = max(r for r, _c in cells)
        cells = [(max_r - r, c) for r, c in cells]
    if rng.random() < 0.5:
        cells = [(c, r) for r, c in cells]

    cells = list(_normalize_cells(cells))
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _padded_free(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_containers(
    rng: random.Random,
    H: int,
    left_w: int,
    specs: list[dict],
) -> list[tuple[int, int, int, int]] | None:
    occ = np.zeros((H, left_w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)
    boxes = [None] * len(specs)
    for idx in order:
        oh = specs[idx]["outer_h"]
        ow = specs[idx]["outer_w"]
        candidates = []
        for r0 in range(1, H - oh):
            for c0 in range(1, left_w - ow):
                if _padded_free(occ, r0, c0, oh, ow, pad=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        occ[r0 : r0 + oh, c0 : c0 + ow] = True
        boxes[idx] = (r0, c0, r0 + oh - 1, c0 + ow - 1)
    return boxes


def _stamp_container(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_fragments(
    rng: random.Random,
    grid: np.ndarray,
    sep_col: int,
    fragments: list[dict],
) -> bool:
    H, W = grid.shape
    occ = np.zeros_like(grid, dtype=bool)
    occ[:, : sep_col + 1] = True
    used_tops = defaultdict(set)
    order = list(range(len(fragments)))
    rng.shuffle(order)

    for idx in order:
        frag = fragments[idx]
        color = frag["color"]
        mask = frag["mask"]
        mh, mw = mask.shape
        candidates = []
        for r0 in range(1, H - mh):
            if r0 in used_tops[color]:
                continue
            for c0 in range(sep_col + 2, W - mw):
                if _padded_free(occ, r0, c0, mh, mw, pad=1):
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        grid[r0 : r0 + mh, c0 : c0 + mw][mask] = color
        occ[r0 : r0 + mh, c0 : c0 + mw][mask] = True
        used_tops[color].add(r0)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(15, 30)
    W = rng.randint(18, 30)
    left_w_min = 8
    left_w_max = min(15, W - 8)
    if left_w_min > left_w_max:
        return None
    left_w = rng.randint(left_w_min, left_w_max)
    right_w = W - left_w - 1
    if right_w < 7:
        return None

    n_colors = rng.randint(2, min(4, len(COLOR_POOL)))
    colors = rng.sample(COLOR_POOL, n_colors)

    counts = [rng.randint(1, 3 if H >= 22 else 2) for _ in range(n_colors)]
    if all(count == 1 for count in counts):
        counts[rng.randrange(n_colors)] = 2

    specs = []
    all_fragments = []
    for color, count in zip(colors, counts):
        masks = [_random_shape(rng) for _ in range(count)]
        stack_h = sum(mask.shape[0] for mask in masks)
        max_w = max(mask.shape[1] for mask in masks)
        if max_w > left_w - 4:
            return None

        min_inner_h = max(3, stack_h)
        max_inner_h = min(H - 4, stack_h + 4)
        if min_inner_h > max_inner_h:
            return None
        inner_h = rng.randint(min_inner_h, max_inner_h)

        max_inner_w = min(left_w - 4, max(5, max_w + 4))
        if max_w > max_inner_w:
            return None
        inner_w = rng.randint(max(3, max_w), max_inner_w)

        specs.append(
            {
                "color": color,
                "masks": masks,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )
        for mask in masks:
            all_fragments.append({"color": color, "mask": mask})

    boxes = _place_containers(rng, H, left_w, specs)
    if boxes is None:
        return None

    grid = np.full((H, W), BG, dtype=int)
    grid[:, left_w] = SEP
    for spec, box in zip(specs, boxes):
        _stamp_container(grid, box, spec["color"])

    if not _place_fragments(rng, grid, left_w, all_fragments):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-stacking puzzle")
