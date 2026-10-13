from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIR_TO_OFFSET = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}

QUADRANTS = ("TL", "TR", "BL", "BR")
IDENTITY_DIRECTIONS = {
    "TL": "NW",
    "TR": "NE",
    "BL": "SW",
    "BR": "SE",
}

BASE_SHAPES = {
    2: [
        ((0, 0), (0, 1)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ],
    5: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ],
}

GEN_RETRIES = 500
_GEN_SUCCESS_COUNT = 0


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize_shape(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in cells]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _transform_shape(
    cells: Iterable[tuple[int, int]],
    rotation: int,
    reflect: bool,
) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in cells]
    for _ in range(rotation % 4):
        pts = [(c, -r) for r, c in pts]
    if reflect:
        pts = [(r, -c) for r, c in pts]
    return _normalize_shape(pts)


def _shape_variants() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    out: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for area, bases in BASE_SHAPES.items():
        seen: set[tuple[tuple[int, int], ...]] = set()
        variants: list[tuple[tuple[int, int], ...]] = []
        for base in bases:
            for rotation in range(4):
                for reflect in (False, True):
                    shape = _transform_shape(base, rotation, reflect)
                    if shape not in seen:
                        seen.add(shape)
                        variants.append(shape)
        out[area] = variants
    return out


SHAPES_BY_AREA = _shape_variants()


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _quadrant_slices(
    h: int,
    w: int,
    sep_r: int,
    sep_c: int,
) -> dict[str, tuple[int, int, int, int]]:
    return {
        "TL": (0, sep_r, 0, sep_c),
        "TR": (0, sep_r, sep_c + 1, w),
        "BL": (sep_r + 1, h, 0, sep_c),
        "BR": (sep_r + 1, h, sep_c + 1, w),
    }


def _direction_from_centroid(
    centroid_r: float,
    centroid_c: float,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> str:
    center_r = (int(r0) + int(r1) - 1) / 2.0
    center_c = (int(c0) + int(c1) - 1) / 2.0
    eps = 1e-9
    if abs(centroid_r - center_r) <= eps or abs(centroid_c - center_c) <= eps:
        raise AssertionError("ambiguous centroid direction")
    ns = "N" if centroid_r < center_r else "S"
    ew = "W" if centroid_c < center_c else "E"
    return ns + ew


def _extract_object(
    grid: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> dict:
    patch = grid[r0:r1, c0:c1]
    mask = patch != 0
    coords = np.argwhere(mask)
    assert coords.size > 0, "empty quadrant"
    colors = {int(v) for v in patch[mask]}
    assert len(colors) == 1, "quadrant must contain exactly one foreground color"
    assert _is_single_component(mask), "quadrant foreground must be one 4-connected object"

    area = int(coords.shape[0])
    centroid_r = float(r0 + coords[:, 0].mean())
    centroid_c = float(c0 + coords[:, 1].mean())
    direction = _direction_from_centroid(centroid_r, centroid_c, r0, r1, c0, c1)
    return {
        "color": int(next(iter(colors))),
        "area": area,
        "centroid": (centroid_r, centroid_c),
        "direction": direction,
        "bbox": (
            int(r0 + coords[:, 0].min()),
            int(c0 + coords[:, 1].min()),
            int(r0 + coords[:, 0].max()),
            int(c0 + coords[:, 1].max()),
        ),
    }


def _parse_grid(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "grid must have odd height and width"
    sep_r, sep_c = h // 2, w // 2
    assert np.all(g[sep_r, :] == 0), "center separator row must be black"
    assert np.all(g[:, sep_c] == 0), "center separator column must be black"

    infos: list[dict] = []
    colors = set()
    directions = set()
    for label, (r0, r1, c0, c1) in _quadrant_slices(h, w, sep_r, sep_c).items():
        info = _extract_object(g, r0, r1, c0, c1)
        br0, bc0, br1, bc1 = info["bbox"]
        if label.startswith("T"):
            assert br1 <= sep_r - 2, "top object touches separator row"
        else:
            assert br0 >= sep_r + 2, "bottom object touches separator row"
        if label.endswith("L"):
            assert bc1 <= sep_c - 2, "left object touches separator column"
        else:
            assert bc0 >= sep_c + 2, "right object touches separator column"
        info["label"] = label
        infos.append(info)
        colors.add(info["color"])
        directions.add(info["direction"])

    assert len(colors) == 4, "quadrant colors must be distinct"
    assert len(directions) == 4, "quadrant directions must be a permutation of NW/NE/SW/SE"
    return sep_r, sep_c, infos


def _outlier_index(areas: list[int]) -> int:
    scores = [sum(abs(a - b) for b in areas) for a in areas]
    best = max(scores)
    idxs = [i for i, score in enumerate(scores) if score == best]
    assert len(idxs) == 1, "area outlier must be unique"
    return int(idxs[0])


def _render_expected(grid: np.ndarray, sep_r: int, sep_c: int, infos: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    outlier = infos[_outlier_index([info["area"] for info in infos])]
    fill_color = int(outlier["color"])

    for c in range(out.shape[1]):
        if c != sep_c:
            out[sep_r, c] = fill_color
    for r in range(out.shape[0]):
        if r != sep_r:
            out[r, sep_c] = fill_color

    assert out[sep_r, sep_c] == 0
    for info in infos:
        dr, dc = DIR_TO_OFFSET[info["direction"]]
        rr, cc = sep_r + dr, sep_c + dc
        assert grid[rr, cc] == 0, "diagonal marker cell must be empty in input"
        out[rr, cc] = int(info["color"])
    return out


def _render_identity_shortcut(grid: np.ndarray, sep_r: int, sep_c: int, infos: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    fill_color = int(infos[_outlier_index([info["area"] for info in infos])]["color"])
    for c in range(out.shape[1]):
        if c != sep_c:
            out[sep_r, c] = fill_color
    for r in range(out.shape[0]):
        if r != sep_r:
            out[r, sep_c] = fill_color
    for info in infos:
        dr, dc = DIR_TO_OFFSET[IDENTITY_DIRECTIONS[info["label"]]]
        out[sep_r + dr, sep_c + dc] = int(info["color"])
    return out


def _render_area_shortcut(
    grid: np.ndarray,
    sep_r: int,
    sep_c: int,
    infos: list[dict],
    choose: str,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    areas = [info["area"] for info in infos]
    if choose == "largest":
        idx = max(range(4), key=lambda i: areas[i])
    elif choose == "smallest":
        idx = min(range(4), key=lambda i: areas[i])
    else:
        raise ValueError("unknown shortcut")
    fill_color = int(infos[idx]["color"])
    for c in range(out.shape[1]):
        if c != sep_c:
            out[sep_r, c] = fill_color
    for r in range(out.shape[0]):
        if r != sep_r:
            out[r, sep_c] = fill_color
    for info in infos:
        dr, dc = DIR_TO_OFFSET[info["direction"]]
        out[sep_r + dr, sep_c + dc] = int(info["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color the center-cross diagonals by quadrant-relative direction and fill the separator cross with the area outlier's color.

    Input structure:
    - The grid has odd height and width, with a single black separator row and a
      single black separator column crossing at the center and splitting the grid
      into four equal quadrants.
    - Each quadrant contains exactly one small 4-connected foreground object of a
      distinct color, and each object stays at least one cell away from the
      separator row and separator column.
    - The four objects occupy distinct relative directions within their own
      quadrants: one NW, one NE, one SW, and one SE when judged by centroid
      against the quadrant center.

    Rule:
    - Keep the four quadrant objects unchanged.
    - For each quadrant object, compute its centroid direction relative to its
      quadrant center and paint that object's color onto the matching diagonal
      cell around the central intersection.
    - Compare the four object areas, find the unique area outlier, and recolor
      every separator-row and separator-column cell except the center with that
      outlier object's color.

    Key insight:
    - The same quadrant objects encode two independent signals at once: centroid
      direction determines which diagonal marker gets each color, while area
      comparison determines the separator-cross fill color.
    """
    sep_r, sep_c, infos = _parse_grid(np.asarray(grid, dtype=int))
    return _render_expected(np.asarray(grid, dtype=int), sep_r, sep_c, infos)


def _quadrant_position_candidates(
    shape: tuple[tuple[int, int], ...],
    label: str,
    quad_h: int,
    quad_w: int,
    target_direction: str,
) -> list[tuple[int, int]]:
    shape_h, shape_w = _shape_size(shape)
    row_lo = 0 if label.startswith("T") else 1
    row_hi = quad_h - shape_h - (1 if label.startswith("T") else 0)
    col_lo = 0 if label.endswith("L") else 1
    col_hi = quad_w - shape_w - (1 if label.endswith("L") else 0)
    if row_hi < row_lo or col_hi < col_lo:
        return []

    center_r = (quad_h - 1) / 2.0
    center_c = (quad_w - 1) / 2.0
    local_rs = np.array([r for r, _ in shape], dtype=float)
    local_cs = np.array([c for _, c in shape], dtype=float)

    candidates: list[tuple[int, int]] = []
    for top in range(row_lo, row_hi + 1):
        centroid_r = float(top + local_rs.mean())
        if abs(centroid_r - center_r) <= 1e-9:
            continue
        ns = "N" if centroid_r < center_r else "S"
        for left in range(col_lo, col_hi + 1):
            centroid_c = float(left + local_cs.mean())
            if abs(centroid_c - center_c) <= 1e-9:
                continue
            ew = "W" if centroid_c < center_c else "E"
            if ns + ew == target_direction:
                candidates.append((top, left))
    return candidates


def _choose_area_profile(mode: str) -> tuple[list[int], int]:
    if mode == "high":
        profiles = [
            ([2, 2, 2], 4),
            ([2, 2, 2], 5),
            ([3, 3, 3], 5),
            ([2, 2, 3], 5),
        ]
    else:
        profiles = [
            ([4, 4, 4], 2),
            ([5, 5, 5], 3),
            ([4, 4, 5], 2),
        ]
    return random.choice(profiles)


def _sample_assignment(values: list[int], outlier: int) -> list[int]:
    outlier_idx = random.randrange(4)
    areas = [0, 0, 0, 0]
    areas[outlier_idx] = int(outlier)
    others = list(values)
    random.shuffle(others)
    write_idx = 0
    for i in range(4):
        if i == outlier_idx:
            continue
        areas[i] = int(others[write_idx])
        write_idx += 1
    return areas


def _direction_assignment() -> list[str]:
    while True:
        dirs = random.sample(list(DIR_TO_OFFSET), 4)
        fixed = sum(
            int(dirs[i] == IDENTITY_DIRECTIONS[QUADRANTS[i]])
            for i in range(4)
        )
        if fixed <= 1:
            return dirs


def _place_quadrant_object(
    grid: np.ndarray,
    label: str,
    color: int,
    area: int,
    direction: str,
    sep_r: int,
    sep_c: int,
) -> bool:
    h, w = grid.shape
    slices = _quadrant_slices(h, w, sep_r, sep_c)[label]
    r0, r1, c0, c1 = slices
    quad_h, quad_w = r1 - r0, c1 - c0

    variants = list(SHAPES_BY_AREA[int(area)])
    random.shuffle(variants)
    for shape in variants:
        candidates = _quadrant_position_candidates(shape, label, quad_h, quad_w, direction)
        if not candidates:
            continue
        top, left = random.choice(candidates)
        for dr, dc in shape:
            grid[r0 + top + dr, c0 + left + dc] = int(color)
        return True
    return False


def _semantic_checks(
    grid: np.ndarray,
    intended_dirs: list[str],
    intended_areas: list[int],
    mode: str,
) -> bool:
    try:
        sep_r, sep_c, infos = _parse_grid(grid)
    except AssertionError:
        return False

    by_label = {info["label"]: info for info in infos}
    if [by_label[label]["direction"] for label in QUADRANTS] != intended_dirs:
        return False
    if [by_label[label]["area"] for label in QUADRANTS] != intended_areas:
        return False

    expected = _render_expected(grid, sep_r, sep_c, infos)
    if not np.array_equal(main(grid), expected):
        return False

    if np.array_equal(_render_identity_shortcut(grid, sep_r, sep_c, infos), expected):
        return False

    wrong_area = "smallest" if mode == "high" else "largest"
    if np.array_equal(_render_area_shortcut(grid, sep_r, sep_c, infos, wrong_area), expected):
        return False

    diag_colors = {
        int(expected[sep_r + dr, sep_c + dc])
        for dr, dc in DIR_TO_OFFSET.values()
    }
    if len(diag_colors) != 4:
        return False

    fill_color = int(expected[sep_r, 0 if sep_c != 0 else 1])
    if fill_color not in {info["color"] for info in infos}:
        return False
    return True


def _generate_attempt(mode: str) -> np.ndarray | None:
    height = random.choice([11, 13, 15, 17, 19])
    width = random.choice([11, 13, 15, 17, 19])
    sep_r, sep_c = height // 2, width // 2
    grid = np.zeros((height, width), dtype=int)

    normal_areas, outlier_area = _choose_area_profile(mode)
    assigned_areas = _sample_assignment(normal_areas, outlier_area)
    assigned_dirs = _direction_assignment()
    assigned_colors = random.sample([1, 2, 3, 4], 4)

    if min(sep_r, sep_c) < 5:
        return None

    for label, color, area, direction in zip(
        QUADRANTS,
        assigned_colors,
        assigned_areas,
        assigned_dirs,
    ):
        if not _place_quadrant_object(grid, label, color, area, direction, sep_r, sep_c):
            return None

    if not _semantic_checks(grid, assigned_dirs, assigned_areas, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_SUCCESS_COUNT
    mode = "high" if _GEN_SUCCESS_COUNT % 2 == 0 else "low"
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(mode)
        if grid is not None:
            _GEN_SUCCESS_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
