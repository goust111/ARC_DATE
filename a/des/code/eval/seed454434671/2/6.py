from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 7)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
MAX_GENERATION_ATTEMPTS = 400


def _mode(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _find_separator_row(g: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = [
        (r, int(g[r, 0]))
        for r in range(g.shape[0])
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != int(bg)
    ]
    assert len(candidates) == 1, "expected exactly one full non-background separator row"
    return candidates[0]


def _is_connected(mask: np.ndarray, directions: tuple[tuple[int, int], ...]) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = [tuple(map(int, coords[0]))]
    seen[stack[0]] = True
    count = 0

    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                stack.append((int(nr), int(nc)))

    return count == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _extract_components(
    region: np.ndarray,
    row_offset: int,
    ignored: set[int],
) -> list[dict[str, object]]:
    h, w = region.shape
    seen = np.zeros_like(region, dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if color in ignored or bool(seen[r, c]):
                continue

            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR8:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and int(region[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            local_coords = np.array(coords, dtype=int)
            r0 = int(local_coords[:, 0].min())
            c0 = int(local_coords[:, 1].min())
            r1 = int(local_coords[:, 0].max())
            c1 = int(local_coords[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[local_coords[:, 0] - r0, local_coords[:, 1] - c0] = True

            global_coords = local_coords.copy()
            global_coords[:, 0] += int(row_offset)
            components.append(
                {
                    "color": color,
                    "bbox": (r0 + row_offset, c0, r1 + row_offset, c1),
                    "coords": global_coords,
                    "mask": mask,
                }
            )

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))  # type: ignore[index]
    return components


def _masks_complementary(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, np.logical_not(b))


def _match_components(
    upper: list[dict[str, object]],
    lower: list[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    group_keys = {
        (int(comp["color"]), *comp["mask"].shape)  # type: ignore[index, union-attr]
        for comp in upper + lower
    }
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []

    for key in sorted(group_keys):
        group_upper = [comp for comp in upper if (int(comp["color"]), *comp["mask"].shape) == key]  # type: ignore[index, union-attr]
        group_lower = [comp for comp in lower if (int(comp["color"]), *comp["mask"].shape) == key]  # type: ignore[index, union-attr]
        if not group_upper or not group_lower:
            continue

        adjacency: list[list[int]] = []
        for u_comp in group_upper:
            u_mask = u_comp["mask"]  # type: ignore[index]
            neighbors = [
                j
                for j, l_comp in enumerate(group_lower)
                if _masks_complementary(u_mask, l_comp["mask"])  # type: ignore[index]
            ]
            adjacency.append(neighbors)

        matched_lower = [-1] * len(group_lower)

        def dfs(u_idx: int, seen_lower: list[bool]) -> bool:
            for l_idx in adjacency[u_idx]:
                if seen_lower[l_idx]:
                    continue
                seen_lower[l_idx] = True
                if matched_lower[l_idx] == -1 or dfs(matched_lower[l_idx], seen_lower):
                    matched_lower[l_idx] = u_idx
                    return True
            return False

        for u_idx in range(len(group_upper)):
            dfs(u_idx, [False] * len(group_lower))

        for l_idx, u_idx in enumerate(matched_lower):
            if u_idx != -1:
                pairs.append((group_upper[u_idx], group_lower[l_idx]))

    pairs.sort(key=lambda item: (item[1]["bbox"][0], item[1]["bbox"][1], item[0]["bbox"][0], item[0]["bbox"][1]))  # type: ignore[index]
    return pairs


def _pair_signatures(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]]:
    bg = _mode([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])])
    sep_row, sep_color = _find_separator_row(grid, bg)
    upper = _extract_components(grid[:sep_row, :], 0, {bg, sep_color})
    lower = _extract_components(grid[sep_row + 1 :, :], sep_row + 1, {bg, sep_color})
    return [
        (int(u["color"]), tuple(map(int, u["bbox"])), tuple(map(int, l["bbox"])))
        for u, l in _match_components(upper, lower)
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair upper and lower fragments by same-color geometric complement, erase the matched
    upper fragment, and replace the matched lower fragment with its completed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])])
    sep_row, sep_color = _find_separator_row(g, bg)
    upper = _extract_components(g[:sep_row, :], 0, {bg, sep_color})
    lower = _extract_components(g[sep_row + 1 :, :], sep_row + 1, {bg, sep_color})
    matches = _match_components(upper, lower)

    out = g.copy()
    for upper_comp, lower_comp in matches:
        upper_coords = upper_comp["coords"]  # type: ignore[index]
        out[upper_coords[:, 0], upper_coords[:, 1]] = bg
        r0, c0, r1, c1 = lower_comp["bbox"]  # type: ignore[index]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(lower_comp["color"])

    out[sep_row, :] = sep_color
    return out


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _sample_complementary_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    for _ in range(200):
        fill = random.uniform(0.35, 0.65)
        mask = np.random.random((height, width)) < fill
        pixels = int(mask.sum())
        if pixels < 2 or pixels > area - 2:
            continue
        if not _touches_all_sides(mask) or not _touches_all_sides(~mask):
            continue
        if not _is_connected(mask, DIR8) or not _is_connected(~mask, DIR8):
            continue
        return mask
    return None


def _sample_distractor_mask(height: int, width: int) -> np.ndarray | None:
    if random.random() < 0.25:
        return np.ones((height, width), dtype=bool)

    dirs = DIR4 if random.random() < 0.5 else DIR8
    for _ in range(120):
        fill = random.uniform(0.35, 0.8)
        mask = np.random.random((height, width)) < fill
        if int(mask.sum()) == 0:
            continue
        if not _touches_all_sides(mask):
            continue
        if _is_connected(mask, dirs):
            return mask
    return None


def _place_mask(
    canvas: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    mask: np.ndarray,
    color: int,
) -> dict[str, object] | None:
    height, width = canvas.shape
    ph, pw = mask.shape
    if ph + 2 > height or pw + 2 > width:
        return None

    min_r = 1
    max_r = height - ph - 1
    min_c = 1
    max_c = width - pw - 1
    if max_r < min_r or max_c < min_c:
        return None

    for _ in range(160):
        r0 = random.randint(min_r, max_r)
        c0 = random.randint(min_c, max_c)
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if any(_boxes_overlap(bbox, other, gap=1) for other in boxes):
            continue

        patch = canvas[r0 : r0 + ph, c0 : c0 + pw]
        patch[mask] = int(color)
        boxes.append(bbox)
        coords = np.argwhere(mask).astype(int)
        coords[:, 0] += r0
        coords[:, 1] += c0
        return {"bbox": bbox, "coords": coords, "mask": mask, "color": int(color)}

    return None


def _build_expected(
    grid: np.ndarray,
    sep_row: int,
    matches: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    row_offset = sep_row + 1
    for spec in matches:
        upper_coords = spec["upper_coords"]  # type: ignore[index]
        out[upper_coords[:, 0], upper_coords[:, 1]] = BG
        r0, c0, r1, c1 = spec["lower_bbox"]  # type: ignore[index]
        out[r0 + row_offset : r1 + row_offset + 1, c0 : c1 + 1] = int(spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(1, 3)
    extra_top = random.randint(0, 2)
    extra_bottom = random.randint(0, 2)
    top_h = random.randint(9, 13)
    bottom_h = random.randint(9, 13)
    width = random.randint(max(18, 14 + 2 * pair_count), 28)

    top = np.full((top_h, width), BG, dtype=int)
    bottom = np.full((bottom_h, width), BG, dtype=int)
    top_boxes: list[tuple[int, int, int, int]] = []
    bottom_boxes: list[tuple[int, int, int, int]] = []
    matched_specs: list[dict[str, object]] = []

    for color in random.sample(PALETTE, pair_count):
        ph = random.randint(3, min(6, top_h - 2, bottom_h - 2))
        pw = random.randint(3, min(6, width - 2))
        mask = _sample_complementary_mask(ph, pw)
        if mask is None:
            return None

        upper_mask = mask if random.random() < 0.5 else ~mask
        lower_mask = ~upper_mask
        upper_info = _place_mask(top, top_boxes, upper_mask, color)
        lower_info = _place_mask(bottom, bottom_boxes, lower_mask, color)
        if upper_info is None or lower_info is None:
            return None

        matched_specs.append(
            {
                "color": int(color),
                "upper_coords": upper_info["coords"],
                "lower_bbox": lower_info["bbox"],
            }
        )

    for _ in range(extra_top):
        ph = random.randint(2, min(5, top_h - 2))
        pw = random.randint(2, min(5, width - 2))
        mask = _sample_distractor_mask(ph, pw)
        if mask is None:
            return None
        if _place_mask(top, top_boxes, mask, random.choice(PALETTE)) is None:
            return None

    for _ in range(extra_bottom):
        ph = random.randint(2, min(5, bottom_h - 2))
        pw = random.randint(2, min(5, width - 2))
        mask = _sample_distractor_mask(ph, pw)
        if mask is None:
            return None
        if _place_mask(bottom, bottom_boxes, mask, random.choice(PALETTE)) is None:
            return None

    grid = np.full((top_h + 1 + bottom_h, width), BG, dtype=int)
    grid[:top_h, :] = top
    grid[top_h, :] = GREY
    grid[top_h + 1 :, :] = bottom

    expected = _build_expected(grid, top_h, matched_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    intended_signatures = sorted(
        (
            int(spec["color"]),
            (
                int(spec["upper_coords"][:, 0].min()),  # type: ignore[index]
                int(spec["upper_coords"][:, 1].min()),  # type: ignore[index]
                int(spec["upper_coords"][:, 0].max()),  # type: ignore[index]
                int(spec["upper_coords"][:, 1].max()),  # type: ignore[index]
            ),
            (
                int(spec["lower_bbox"][0] + top_h + 1),  # type: ignore[index]
                int(spec["lower_bbox"][1]),  # type: ignore[index]
                int(spec["lower_bbox"][2] + top_h + 1),  # type: ignore[index]
                int(spec["lower_bbox"][3]),  # type: ignore[index]
            ),
        )
        for spec in matched_specs
    )
    actual_signatures = sorted(_pair_signatures(grid))
    if actual_signatures != intended_signatures:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
