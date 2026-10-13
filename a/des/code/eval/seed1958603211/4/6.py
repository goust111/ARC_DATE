from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
FRAGMENT_COUNT_CHOICES = [3, 4, 5]
GRID_H_RANGE = (13, 20)
GRID_W_RANGE = (20, 31)
PLACEMENT_GAP = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _path_endpoints(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    pts: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        deg = 0
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                deg += 1
        if deg == 1:
            pts.append((int(r), int(c)))
    return pts


def _extract_fragments(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = _get_bg(g)
    comps = _components(g != bg)

    marker_candidates: list[tuple[int, int]] = []
    fragments: list[dict[str, object]] = []
    for comp in comps:
        if len(comp) == 1 and comp[0][0] == h - 1:
            marker_candidates.append(comp[0])
            continue

        patch = _crop_component(g, comp, bg)
        mask = patch != bg
        endpoints = _path_endpoints(mask)
        assert len(endpoints) == 2, "each fragment must be a simple 2-endpoint path"

        colors = Counter(int(v) for v in patch[mask])
        main_color = max(colors.items(), key=lambda kv: kv[1])[0]
        ports = [
            {"color": int(patch[r, c]), "pos": (int(r), int(c))}
            for r, c in endpoints
        ]
        fragments.append(
            {
                "patch": patch,
                "main_color": int(main_color),
                "ports": ports,
            }
        )

    assert len(marker_candidates) == 1, "expected exactly one bottom-row start marker"
    assert fragments, "expected at least one path fragment"
    mr, mc = marker_candidates[0]
    return int(bg), int(g[mr, mc]), fragments


def _order_fragments(fragments: list[dict[str, object]], start_color: int) -> list[tuple[dict[str, object], int, int]]:
    remaining = set(range(len(fragments)))
    current = int(start_color)
    ordered: list[tuple[dict[str, object], int, int]] = []

    while remaining:
        matches: list[tuple[int, int]] = []
        for idx in remaining:
            ports = fragments[idx]["ports"]
            for port_idx, port in enumerate(ports):
                if int(port["color"]) == current:
                    matches.append((idx, port_idx))
        assert len(matches) == 1, "port colors must define a unique chain"

        idx, entry_idx = matches[0]
        exit_idx = 1 - entry_idx
        entry_color = int(fragments[idx]["ports"][entry_idx]["color"])
        exit_color = int(fragments[idx]["ports"][exit_idx]["color"])
        ordered.append((fragments[idx], entry_color, exit_color))
        remaining.remove(idx)
        current = exit_color

    return ordered


def _orient_fragment(
    fragment: dict[str, object], entry_color: int, exit_color: int, bg: int
) -> dict[str, object]:
    patch = np.asarray(fragment["patch"], dtype=int)
    main_color = int(fragment["main_color"])

    for k in range(4):
        rot = np.rot90(patch, k)
        mask = rot != bg
        endpoints = _path_endpoints(mask)
        if len(endpoints) != 2:
            continue

        by_color = {int(rot[r, c]): (int(r), int(c)) for r, c in endpoints}
        if entry_color not in by_color or exit_color not in by_color:
            continue

        er, ec = by_color[entry_color]
        xr, xc = by_color[exit_color]
        if ec != 0 or xc != rot.shape[1] - 1:
            continue

        clean = rot.copy()
        clean[er, ec] = main_color
        clean[xr, xc] = main_color
        return {
            "patch": clean,
            "entry_row": int(er),
            "exit_row": int(xr),
        }

    raise AssertionError("fragment cannot be rotated into left-to-right orientation")


def _assemble_ordered_fragments(
    oriented: list[dict[str, object]], bg: int
) -> np.ndarray:
    tops: list[int] = []
    current_row: int | None = None
    for idx, piece in enumerate(oriented):
        entry_row = int(piece["entry_row"])
        exit_row = int(piece["exit_row"])
        if idx == 0:
            top = 0
        else:
            assert current_row is not None
            top = int(current_row - entry_row)
        tops.append(top)
        current_row = int(top + exit_row)

    min_top = min(tops)
    adj_tops = [top - min_top for top in tops]
    out_h = max(top + int(piece["patch"].shape[0]) for top, piece in zip(adj_tops, oriented))
    out_w = sum(int(piece["patch"].shape[1]) for piece in oriented)
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for top, piece in zip(adj_tops, oriented):
        patch = np.asarray(piece["patch"], dtype=int)
        mask = patch != bg
        h, w = patch.shape
        out[top : top + h, col : col + w][mask] = patch[mask]
        col += w

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain path fragments by matching endpoint colors, rotate each fragment so the
    matched entry is on the left and the exit is on the right, then assemble the
    cleaned fragments into one continuous horizontal path.
    """
    g = np.asarray(grid, dtype=int)
    bg, start_color, fragments = _extract_fragments(g)
    ordered = _order_fragments(fragments, start_color)
    oriented = [
        _orient_fragment(fragment, entry_color, exit_color, bg)
        for fragment, entry_color, exit_color in ordered
    ]
    return _assemble_ordered_fragments(oriented, bg)


CANONICAL_TEMPLATES = [
    {
        "shape": (3, 4),
        "path": [(1, 0), (1, 1), (1, 2), (1, 3)],
    },
    {
        "shape": (3, 4),
        "path": [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (1, 3)],
    },
    {
        "shape": (3, 4),
        "path": [(1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (1, 3)],
    },
    {
        "shape": (3, 5),
        "path": [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
    },
    {
        "shape": (3, 5),
        "path": [(1, 0), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3), (1, 4)],
    },
    {
        "shape": (3, 5),
        "path": [(1, 0), (1, 1), (2, 1), (2, 2), (2, 3), (1, 3), (1, 4)],
    },
    {
        "shape": (3, 5),
        "path": [(1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (1, 3), (1, 4)],
    },
    {
        "shape": (3, 5),
        "path": [(1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3), (1, 4)],
    },
]


def _render_piece(
    template: dict[str, object], bg: int, main_color: int, entry_color: int, exit_color: int
) -> tuple[np.ndarray, np.ndarray]:
    h, w = template["shape"]
    path = template["path"]
    patch = np.full((h, w), bg, dtype=int)
    for r, c in path:
        patch[r, c] = main_color
    er, ec = path[0]
    xr, xc = path[-1]
    patch[er, ec] = entry_color
    patch[xr, xc] = exit_color
    clean = patch.copy()
    clean[er, ec] = main_color
    clean[xr, xc] = main_color
    return patch, clean


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_fragments(
    rng: random.Random, grid: np.ndarray, patches: list[np.ndarray]
) -> bool:
    h, w = grid.shape
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if h - ph - 3 < 1 or w - pw - 2 < 1:
            return False

        placed = False
        for _ in range(200):
            r0 = rng.randint(1, h - ph - 3)
            c0 = rng.randint(1, w - pw - 2)
            rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_rect_overlaps(rect, prev, PLACEMENT_GAP) for prev in rects):
                continue
            mask = patch != _get_bg(grid)
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
            rects.append(rect)
            placed = True
            break
        if not placed:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice(FRAGMENT_COUNT_CHOICES)
    bg = rng.randrange(10)
    pool = [c for c in range(10) if c != bg]
    rng.shuffle(pool)

    path_palette = pool[:3]
    port_palette = pool[3 : 3 + count + 1]
    if len(port_palette) != count + 1:
        return None

    height = rng.randint(*GRID_H_RANGE)
    min_width = max(GRID_W_RANGE[0], 6 * count)
    width = rng.randint(min_width, GRID_W_RANGE[1] - 1) if min_width < GRID_W_RANGE[1] else min_width
    if width >= GRID_W_RANGE[1]:
        width = GRID_W_RANGE[1] - 1
    if width < min_width:
        return None

    main_colors = [path_palette[i % len(path_palette)] for i in range(count)]
    rng.shuffle(main_colors)
    if count >= 2 and len(set(main_colors)) == 1:
        main_colors[1] = path_palette[1]

    canonical_clean: list[np.ndarray] = []
    input_patches: list[np.ndarray] = []
    template_ids = list(range(len(CANONICAL_TEMPLATES)))
    rng.shuffle(template_ids)

    for idx in range(count):
        template = CANONICAL_TEMPLATES[template_ids[idx % len(template_ids)]]
        patch, clean = _render_piece(
            template,
            bg,
            main_colors[idx],
            port_palette[idx],
            port_palette[idx + 1],
        )
        rot = rng.randrange(4)
        input_patches.append(np.rot90(patch, rot))
        canonical_clean.append(clean)

    grid = np.full((height, width), bg, dtype=int)
    if not _place_fragments(rng, grid, input_patches):
        return None

    marker_color = port_palette[0]
    marker_cols = list(range(1, width - 1))
    rng.shuffle(marker_cols)
    marker_col = None
    for col in marker_cols:
        if grid[height - 1, col] == bg and grid[height - 2, col] == bg:
            marker_col = col
            break
    if marker_col is None:
        return None
    grid[height - 1, marker_col] = marker_color

    expected = np.concatenate(canonical_clean, axis=1)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
