from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg


BG = 0
CONTAINER_COLOR = 7
MARKER_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 400
AREA_TO_DIMS = {
    1: [(2, 3)],
    2: [(2, 4), (3, 3)],
    3: [(2, 5), (4, 3)],
    4: [(3, 4)],
    6: [(3, 5), (4, 4)],
    9: [(4, 5)],
}
COUNT_PATTERNS = [
    (2, 1, 1),
    (3, 1, 1),
    (3, 2),
    (4, 1, 1),
]


def _make_container_patch(height: int, width: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[:, 0] = CONTAINER_COLOR
    patch[:, -1] = CONTAINER_COLOR
    patch[-1, :] = CONTAINER_COLOR
    return patch


def _extract_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for coords in components_mask(grid == CONTAINER_COLOR, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = crop_bbox(grid, (r0, c0, r1, c1))
        h, w = patch.shape
        assert 2 <= h <= 4 and 3 <= w <= 5, "each container must have valid U dimensions"
        assert np.all(patch[:, 0] == CONTAINER_COLOR)
        assert np.all(patch[:, -1] == CONTAINER_COLOR)
        assert np.all(patch[-1, :] == CONTAINER_COLOR)
        assert np.all(patch[:-1, 1:-1] == BG), "container cavity must be empty"

        skeleton = np.zeros_like(patch)
        skeleton[:, 0] = CONTAINER_COLOR
        skeleton[:, -1] = CONTAINER_COLOR
        skeleton[-1, :] = CONTAINER_COLOR
        assert np.array_equal(patch, skeleton), "orange component must be a plain U shape"

        containers.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "height": int(h),
                "width": int(w),
                "area": int((h - 1) * (w - 2)),
            }
        )

    containers.sort(key=lambda item: (item["c0"], item["r0"]))
    for left, right in zip(containers, containers[1:]):
        assert left["c1"] < right["c0"], "container spans must not overlap"
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop top-row markers into orange containers and fill only the cavity whose area matches the marker color's global count.

    Input structure:
    - The background is black, and the top row contains 4 to 6 single-pixel markers
      colored blue, red, green, or yellow; repeated colors are allowed.
    - The lower half contains 3 or 4 non-overlapping orange U-shaped containers.
    - Each container has an empty rectangular cavity with a unique area.

    Rule:
    - Erase all top-row markers.
    - For each marker, count how many top-row markers share its color.
    - If the marker's column lies inside a container span and that count equals the
      container cavity area, fill the whole cavity with the marker color.
    - If the marker hits a container but the count does not match, discard it.
    - If the marker column hits no container span, place that marker on the bottom
      row in the same column. Keep every orange container wall unchanged.

    Key insight:
    The falling column chooses the candidate container, but the actual fill is gated
    by a global same-color count that must equal that container's cavity area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    top_cols = [int(c) for c in np.flatnonzero(g[0] != bg)]
    counts = Counter(int(g[0, c]) for c in top_cols)
    containers = _extract_containers(g)
    assert containers, "expected at least one container"

    out = g.copy()
    out[0, top_cols] = bg

    fills: dict[int, int] = {}
    for c in top_cols:
        color = int(g[0, c])
        count = int(counts[color])
        hits = [idx for idx, box in enumerate(containers) if box["c0"] <= c <= box["c1"]]
        assert len(hits) <= 1, "container spans must not overlap"
        if not hits:
            out[-1, c] = color
            continue

        idx = int(hits[0])
        if count == int(containers[idx]["area"]):
            prev = fills.get(idx)
            assert prev in (None, color), "a container should not be matched by two colors"
            fills[idx] = color

    for idx, color in fills.items():
        box = containers[int(idx)]
        out[box["r0"] : box["r1"], box["c0"] + 1 : box["c1"]] = int(color)
    return out


def _distribute_extra(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    random.shuffle(parts)
    return parts


def _choose_areas(success_area: int, other_counts: list[int], n_containers: int) -> list[int]:
    available = [area for area in AREA_TO_DIMS if area != success_area]
    safe = [area for area in available if area not in set(other_counts)]
    random.shuffle(safe)
    random.shuffle(available)
    chosen = [success_area]
    for area in safe + available:
        if area not in chosen:
            chosen.append(int(area))
        if len(chosen) == n_containers:
            break
    if len(chosen) != n_containers:
        raise ValueError("failed to choose unique container areas")
    random.shuffle(chosen)
    return chosen


def _build_containers() -> tuple[int, int, list[dict], int]:
    pattern = random.choice(COUNT_PATTERNS)
    success_area = int(pattern[0])
    other_counts = list(pattern[1:])
    n_containers = random.randint(3, 4)
    areas = _choose_areas(success_area, other_counts, n_containers)

    containers: list[dict] = []
    widths: list[int] = []
    heights: list[int] = []
    for area in areas:
        height, width = random.choice(AREA_TO_DIMS[int(area)])
        widths.append(int(width))
        heights.append(int(height))
        containers.append({"area": int(area), "height": int(height), "width": int(width)})

    min_width = sum(widths) + (n_containers + 1)
    max_width = min(20, min_width + 6)
    if min_width > max_width:
        raise ValueError("container layout too wide")
    grid_w = random.randint(max(13, min_width), max_width)

    min_height = max(12, max(heights) + 8)
    max_height = min(18, min_height + 4)
    if min_height > max_height:
        raise ValueError("container layout too tall")
    grid_h = random.randint(min_height, max_height)

    spare = grid_w - sum(widths) - (n_containers + 1)
    gaps = [1] * (n_containers + 1)
    if spare > 0:
        for idx, extra in enumerate(_distribute_extra(spare, n_containers + 1)):
            gaps[idx] += int(extra) - 1

    c = gaps[0]
    for idx, box in enumerate(containers):
        h = int(box["height"])
        w = int(box["width"])
        r0_min = grid_h // 2
        r0_max = grid_h - h - 2
        if r0_min > r0_max:
            raise ValueError("not enough vertical room for lower-half containers")
        r0 = random.randint(r0_min, r0_max)
        box["r0"] = int(r0)
        box["c0"] = int(c)
        box["r1"] = int(r0 + h - 1)
        box["c1"] = int(c + w - 1)
        box["success"] = False
        c += w + gaps[idx + 1]

    success_idx = next(i for i, box in enumerate(containers) if box["area"] == success_area)
    containers[success_idx]["success"] = True
    return grid_h, grid_w, containers, success_idx


def _unused_columns(pool: list[int], used: set[int], count: int) -> list[int] | None:
    avail = [c for c in pool if c not in used]
    if len(avail) < count:
        return None
    chosen = random.sample(avail, count)
    used.update(chosen)
    return [int(c) for c in chosen]


def _build_markers(containers: list[dict], success_idx: int, grid_w: int) -> list[tuple[int, int]]:
    pattern = random.choice([p for p in COUNT_PATTERNS if p[0] == containers[success_idx]["area"]])
    colors = random.sample(MARKER_COLORS, len(pattern))
    counts_by_color = {int(color): int(count) for color, count in zip(colors, pattern)}
    success_color = int(colors[0])

    inside_cols = {idx: list(range(box["c0"], box["c1"] + 1)) for idx, box in enumerate(containers)}
    covered = set()
    for box in containers:
        covered.update(range(box["c0"], box["c1"] + 1))
    outside_cols = [c for c in range(grid_w) if c not in covered]

    used_cols: set[int] = set()
    markers: list[tuple[int, int]] = []

    success_col = _unused_columns(inside_cols[success_idx], used_cols, 1)
    outside_col = _unused_columns(outside_cols, used_cols, 1)
    if success_col is None or outside_col is None:
        raise ValueError("failed to reserve required success columns")
    markers.append((success_col[0], success_color))
    markers.append((outside_col[0], success_color))

    success_count = counts_by_color[success_color]
    mismatch_boxes = [idx for idx, box in enumerate(containers) if idx != success_idx and box["area"] != success_count]
    if len(mismatch_boxes) < max(1, success_count - 2):
        mismatch_boxes = [idx for idx, box in enumerate(containers) if idx != success_idx]
    if len(mismatch_boxes) < max(1, success_count - 2):
        raise ValueError("not enough discard containers for success color")

    for idx in random.sample(mismatch_boxes, success_count - 2):
        col = _unused_columns(inside_cols[idx], used_cols, 1)
        if col is None:
            raise ValueError("failed to place a discard marker")
        markers.append((col[0], success_color))

    need_discard = success_count == 2
    for color in colors[1:]:
        count = counts_by_color[int(color)]
        roles = []
        for _ in range(count):
            if need_discard:
                roles.append("discard")
                need_discard = False
            else:
                roles.append(random.choice(["outside", "discard"]))

        for role in roles:
            if role == "outside":
                col = _unused_columns(outside_cols, used_cols, 1)
                if col is None:
                    role = "discard"
                else:
                    markers.append((col[0], int(color)))
                    continue

            eligible = [
                idx
                for idx, box in enumerate(containers)
                if idx != success_idx and box["area"] != count and any(c not in used_cols for c in inside_cols[idx])
            ]
            if not eligible:
                eligible = [
                    idx
                    for idx, box in enumerate(containers)
                    if box["area"] != count and any(c not in used_cols for c in inside_cols[idx])
                ]
            if not eligible:
                raise ValueError("failed to find a discard placement")
            target = random.choice(eligible)
            col = _unused_columns(inside_cols[target], used_cols, 1)
            if col is None:
                raise ValueError("discard placement ran out of columns")
            markers.append((col[0], int(color)))

    random.shuffle(markers)
    return markers


def _render_input(grid_h: int, grid_w: int, containers: list[dict], markers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for box in containers:
        patch = _make_container_patch(int(box["height"]), int(box["width"]))
        grid[box["r0"] : box["r1"] + 1, box["c0"] : box["c1"] + 1] = patch
    for col, color in markers:
        grid[0, int(col)] = int(color)
    return grid


def _expected_from_specs(grid_h: int, grid_w: int, containers: list[dict], markers: list[tuple[int, int]]) -> np.ndarray:
    out = _render_input(grid_h, grid_w, containers, markers)
    counts = Counter(color for _col, color in markers)
    for col, _color in markers:
        out[0, int(col)] = BG

    fills: dict[int, int] = {}
    for col, color in markers:
        hit = [idx for idx, box in enumerate(containers) if box["c0"] <= col <= box["c1"]]
        if not hit:
            out[-1, int(col)] = int(color)
            continue
        idx = int(hit[0])
        if counts[int(color)] == int(containers[idx]["area"]):
            prev = fills.get(idx)
            if prev not in (None, int(color)):
                raise ValueError("ambiguous container fill in generator")
            fills[idx] = int(color)

    for idx, color in fills.items():
        box = containers[int(idx)]
        out[box["r0"] : box["r1"], box["c0"] + 1 : box["c1"]] = int(color)
    return out


def _validate_semantics(grid: np.ndarray, containers: list[dict], markers: list[tuple[int, int]], success_idx: int) -> None:
    parsed = _extract_containers(grid)
    expected_boxes = sorted((box["r0"], box["c0"], box["r1"], box["c1"], box["area"]) for box in containers)
    actual_boxes = sorted((box["r0"], box["c0"], box["r1"], box["c1"], box["area"]) for box in parsed)
    assert actual_boxes == expected_boxes

    counts = Counter(color for _col, color in markers)
    success_hits = 0
    floor_hits = 0
    discard_hits = 0
    for col, color in markers:
        hit = [idx for idx, box in enumerate(containers) if box["c0"] <= col <= box["c1"]]
        if not hit:
            floor_hits += 1
            continue
        idx = int(hit[0])
        if counts[int(color)] == int(containers[idx]["area"]):
            success_hits += 1
            assert idx == success_idx
        else:
            discard_hits += 1

    assert success_hits >= 1
    assert floor_hits >= 1
    assert discard_hits >= 1
    assert sum(counts.values()) == len(markers)
    assert 4 <= len(markers) <= 6
    assert 3 <= len(containers) <= 4
    assert len({box["area"] for box in containers}) == len(containers)


def _gen_attempt() -> np.ndarray | None:
    try:
        grid_h, grid_w, containers, success_idx = _build_containers()
        markers = _build_markers(containers, success_idx, grid_w)
        if len(markers) < 4 or len(markers) > 6:
            return None

        grid = _render_input(grid_h, grid_w, containers, markers)
        expected = _expected_from_specs(grid_h, grid_w, containers, markers)
        _validate_semantics(grid, containers, markers, success_idx)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return None
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-count-container input")
