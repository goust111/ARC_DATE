from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

TEMPLATE_MASKS = [
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
]

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_split_row(grid: np.ndarray, bg: int) -> int:
    for r in range(1, grid.shape[0]):
        if np.all(grid[r] == bg):
            return r
    raise AssertionError("expected a full background spacer row below the legend")


def _offsets_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r) - 1, int(c) - 1) for r, c in np.argwhere(mask)]


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[int, int, dict[int, list[tuple[int, int]]]]:
    split_row = _find_split_row(grid, bg)
    legend = grid[:split_row]
    coords = np.argwhere(legend != bg)
    assert coords.size, "expected a top legend"

    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    box = legend[r0 : r1 + 1, c0 : c1 + 1]
    wall = int(box[0, 0])

    assert box.shape[0] == 5, "legend height must match the 3x3-cell strip format"
    assert np.all(box[0, :] == wall) and np.all(box[-1, :] == wall), "legend must be framed by grey rows"

    sep_cols = [c for c in range(box.shape[1]) if np.all(box[:, c] == wall)]
    mapping: dict[int, list[tuple[int, int]]] = {}
    for left, right in zip(sep_cols, sep_cols[1:], strict=False):
        if right - left != 4:
            continue
        cell = box[1:4, left + 1 : right]
        if cell.shape != (3, 3):
            continue
        colors = [int(v) for v in np.unique(cell) if int(v) not in (bg, wall)]
        if len(colors) != 1:
            continue
        color = colors[0]
        mapping[color] = _offsets_from_mask(cell == color)

    assert mapping, "legend must yield at least one color->template mapping"
    return split_row, wall, mapping


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _find_regions(body: np.ndarray, wall: int, bg: int) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for comp in _components(body == wall):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        perimeter = _perimeter_cells(r0, c0, r1, c1)
        perim_set = set(perimeter)
        if any(cell not in perim_set for cell in comp):
            continue

        gap_cells = [(r, c) for r, c in perimeter if int(body[r, c]) != wall]
        if len(gap_cells) != 1:
            continue
        if len(comp) != len(perimeter) - 1:
            continue

        gap = gap_cells[0]
        if gap[0] == r0:
            direction = "up"
        elif gap[0] == r1:
            direction = "down"
        elif gap[1] == c0:
            direction = "left"
        else:
            direction = "right"

        interior = body[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere((interior != bg) & (interior != wall))
        if seeds.shape != (1, 2):
            continue
        sr = int(seeds[0, 0]) + r0 + 1
        sc = int(seeds[0, 1]) + c0 + 1
        color = int(body[sr, sc])
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "direction": direction,
                "seed": (sr, sc),
                "color": color,
            }
        )
    return regions


def _fits(anchor: tuple[int, int], offsets: list[tuple[int, int]], interior: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = interior
    for dr, dc in offsets:
        rr = anchor[0] + dr
        cc = anchor[1] + dc
        if rr < ir0 or rr > ir1 or cc < ic0 or cc > ic1:
            return False
    return True


def _apply_growth(
    out: np.ndarray,
    color: int,
    offsets: list[tuple[int, int]],
    seed: tuple[int, int],
    direction: tuple[int, int],
    interior: tuple[int, int, int, int],
    gap: tuple[int, int],
) -> int:
    anchors: list[tuple[int, int]] = []
    anchor = seed
    while _fits(anchor, offsets, interior):
        anchors.append(anchor)
        anchor = (anchor[0] + direction[0], anchor[1] + direction[1])

    assert anchors, "seed placement must admit at least one full template position"

    painted = 0
    for ar, ac in anchors:
        for dr, dc in offsets:
            rr = ar + dr
            cc = ac + dc
            if int(out[rr, cc]) != color:
                painted += 1
            out[rr, cc] = color

    next_anchor = (anchors[-1][0] + direction[0], anchors[-1][1] + direction[1])
    for dr, dc in offsets:
        rr = next_anchor[0] + dr
        cc = next_anchor[1] + dc
        if (rr, cc) == gap:
            orow = rr + direction[0]
            ocol = cc + direction[1]
            if 0 <= orow < out.shape[0] and 0 <= ocol < out.shape[1]:
                if int(out[orow, ocol]) != color:
                    painted += 1
                out[orow, ocol] = color
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top legend strip to map seed colors to 3x3 templates, then grow each template
    through the single missing border cell of its matching grey rectangle. The legend strip is
    removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    split_row, wall, mapping = _parse_legend(g, bg)
    body = g[split_row + 1 :].copy()

    for region in _find_regions(body, wall, bg):
        color = int(region["color"])
        if color not in mapping:
            continue
        r0, c0, r1, c1 = region["bbox"]  # type: ignore[assignment]
        interior = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        _apply_growth(
            body,
            color,
            mapping[color],
            region["seed"],  # type: ignore[arg-type]
            DIRECTIONS[str(region["direction"])],
            interior,
            region["gap"],  # type: ignore[arg-type]
        )

    return body


def _legend_patch(entries: list[tuple[int, np.ndarray]], width: int) -> np.ndarray:
    patch = np.full((5, width), BG, dtype=int)
    legend_w = 4 * len(entries) + 1
    left = random.randint(0, width - legend_w)
    patch[0:5, left] = WALL
    for idx, (color, mask) in enumerate(entries):
        c0 = left + idx * 4
        patch[:, c0] = WALL
        patch[:, c0 + 4] = WALL
        patch[0, c0 : c0 + 5] = WALL
        patch[4, c0 : c0 + 5] = WALL
        cell = patch[1:4, c0 + 1 : c0 + 4]
        cell[mask] = color
    return patch


def _template_extents(offsets: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in offsets]
    cols = [c for _, c in offsets]
    return min(rows), max(rows), min(cols), max(cols)


def _make_region_spec(color: int, mask: np.ndarray, direction_name: str) -> dict[str, object] | None:
    offsets = _offsets_from_mask(mask)
    min_r, max_r, min_c, max_c = _template_extents(offsets)

    for _ in range(100):
        interior_h = random.randint(5, 8)
        interior_w = random.randint(5, 8)
        frontier: list[tuple[int, int]]
        seed_r: int
        seed_c: int
        gap_r: int
        gap_c: int
        full_steps: int

        if direction_name in ("up", "down"):
            max_full = interior_h - max_r + min_r
            anchor_c_lo = 1 - min_c
            anchor_c_hi = interior_w - max_c
            if max_full < 2 or anchor_c_lo > anchor_c_hi:
                continue
            anchor_c = random.randint(anchor_c_lo, anchor_c_hi)
            if direction_name == "up":
                frontier = [off for off in offsets if off[0] == min_r]
                last_anchor_r = 1 - min_r
                full_steps = random.randint(2, min(5, max_full))
                seed_r = last_anchor_r + full_steps - 1
                seed_c = anchor_c
                fr, fc = random.choice(frontier)
                gap_r = 0
                gap_c = anchor_c + fc
            else:
                frontier = [off for off in offsets if off[0] == max_r]
                last_anchor_r = interior_h - max_r
                full_steps = random.randint(2, min(5, max_full))
                seed_r = last_anchor_r - full_steps + 1
                seed_c = anchor_c
                fr, fc = random.choice(frontier)
                gap_r = interior_h + 1
                gap_c = anchor_c + fc
        else:
            max_full = interior_w - max_c + min_c
            anchor_r_lo = 1 - min_r
            anchor_r_hi = interior_h - max_r
            if max_full < 2 or anchor_r_lo > anchor_r_hi:
                continue
            anchor_r = random.randint(anchor_r_lo, anchor_r_hi)
            if direction_name == "left":
                frontier = [off for off in offsets if off[1] == min_c]
                last_anchor_c = 1 - min_c
                full_steps = random.randint(2, min(5, max_full))
                seed_r = anchor_r
                seed_c = last_anchor_c + full_steps - 1
                fr, fc = random.choice(frontier)
                gap_r = anchor_r + fr
                gap_c = 0
            else:
                frontier = [off for off in offsets if off[1] == max_c]
                last_anchor_c = interior_w - max_c
                full_steps = random.randint(2, min(5, max_full))
                seed_r = anchor_r
                seed_c = last_anchor_c - full_steps + 1
                fr, fc = random.choice(frontier)
                gap_r = anchor_r + fr
                gap_c = interior_w + 1

        patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
        patch[0, :] = WALL
        patch[-1, :] = WALL
        patch[:, 0] = WALL
        patch[:, -1] = WALL
        patch[gap_r, gap_c] = BG
        patch[seed_r, seed_c] = color

        interior = (1, 1, interior_h, interior_w)
        if not _fits((seed_r, seed_c), offsets, interior):
            continue
        next_anchor = (seed_r + DIRECTIONS[direction_name][0], seed_c + DIRECTIONS[direction_name][1])
        if not _fits((seed_r, seed_c), offsets, interior):
            continue
        if _fits(next_anchor, offsets, interior) and full_steps == 1:
            continue

        candidate = patch.copy()
        painted = _apply_growth(
            candidate,
            color,
            offsets,
            (seed_r, seed_c),
            DIRECTIONS[direction_name],
            interior,
            (gap_r, gap_c),
        )
        if painted <= len(offsets):
            continue

        return {
            "color": color,
            "mask": mask,
            "offsets": offsets,
            "direction": direction_name,
            "patch": patch,
            "seed": (seed_r, seed_c),
            "gap": (gap_r, gap_c),
            "bbox_h": interior_h + 2,
            "bbox_w": interior_w + 2,
        }
    return None


def _place_regions(
    body_h: int,
    body_w: int,
    specs: list[dict[str, object]],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    body = np.full((body_h, body_w), BG, dtype=int)
    reserved = np.zeros((body_h, body_w), dtype=bool)
    placed: list[dict[str, object]] = []

    order = sorted(specs, key=lambda spec: int(spec["bbox_h"]) * int(spec["bbox_w"]), reverse=True)
    for spec in order:
        bbox_h = int(spec["bbox_h"])
        bbox_w = int(spec["bbox_w"])
        patch = spec["patch"]  # type: ignore[assignment]
        ok = False
        for _ in range(200):
            if body_h - bbox_h - 2 < 1 or body_w - bbox_w - 2 < 1:
                return None
            top = random.randint(1, body_h - bbox_h - 2)
            left = random.randint(1, body_w - bbox_w - 2)
            reserve = reserved[top - 1 : top + bbox_h + 1, left - 1 : left + bbox_w + 1]
            if np.any(reserve):
                continue
            body[top : top + bbox_h, left : left + bbox_w] = patch
            reserved[top - 1 : top + bbox_h + 1, left - 1 : left + bbox_w + 1] = True

            seed_r, seed_c = spec["seed"]  # type: ignore[misc]
            gap_r, gap_c = spec["gap"]  # type: ignore[misc]
            placed.append(
                {
                    "color": int(spec["color"]),
                    "offsets": spec["offsets"],
                    "direction": str(spec["direction"]),
                    "seed": (top + seed_r, left + seed_c),
                    "gap": (top + gap_r, left + gap_c),
                    "bbox": (top, left, top + bbox_h - 1, left + bbox_w - 1),
                }
            )
            ok = True
            break
        if not ok:
            return None
    return body, placed


def _expected_output(body: np.ndarray, regions: list[dict[str, object]]) -> tuple[np.ndarray, int]:
    out = body.copy()
    new_pixels = 0
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]  # type: ignore[misc]
        new_pixels += _apply_growth(
            out,
            int(region["color"]),
            region["offsets"],  # type: ignore[arg-type]
            region["seed"],  # type: ignore[arg-type]
            DIRECTIONS[str(region["direction"])],
            (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            region["gap"],  # type: ignore[arg-type]
        )
    return out, new_pixels


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.randint(2, 4)
    n_legend = random.randint(max(3, n_regions), min(5, len(TEMPLATE_MASKS)))

    colors = random.sample(AVAILABLE_COLORS, n_legend)
    masks = random.sample(TEMPLATE_MASKS, n_legend)
    entries = list(zip(colors, masks, strict=False))

    legend_w = 4 * n_legend + 1
    body_w = max(random.randint(20, 32), legend_w + 2)
    body_h = random.randint(14, 22)

    used_entries = random.sample(entries, n_regions)
    direction_names = random.sample(list(DIRECTIONS.keys()), n_regions)
    specs: list[dict[str, object]] = []
    for (color, mask), direction_name in zip(used_entries, direction_names, strict=False):
        spec = _make_region_spec(color, mask, direction_name)
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_regions(body_h, body_w, specs)
    if placed is None:
        return None
    body, regions = placed

    expected, new_pixels = _expected_output(body, regions)
    if new_pixels < 8:
        return None

    legend = _legend_patch(entries, body_w)
    spacer = np.full((1, body_w), BG, dtype=int)
    inp = np.vstack([legend, spacer, body])

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
