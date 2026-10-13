from __future__ import annotations

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _bbox_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _run_bounds(idxs: np.ndarray) -> tuple[int, int] | None:
    if idxs.size == 0:
        return None
    start = int(idxs[0])
    end = int(idxs[-1])
    if int(idxs.size) != end - start + 1:
        return None
    return start, end


def _detect_container(mask: np.ndarray) -> dict | None:
    h, w = mask.shape
    if h < 4 or w < 4:
        return None
    if bool(mask[1:-1, 1:-1].any()):
        return None

    boundary = np.zeros_like(mask, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    if bool((mask & ~boundary).any()):
        return None

    candidates: list[dict] = []

    top_missing = np.flatnonzero(~mask[0, :])
    top_run = _run_bounds(top_missing)
    if (
        top_run is not None
        and 2 <= int(top_missing.size) <= 4
        and top_run[0] >= 1
        and top_run[1] <= w - 2
        and bool(mask[-1, :].all())
        and bool(mask[:, 0].all())
        and bool(mask[:, -1].all())
    ):
        candidates.append(
            {
                "side": "top",
                "gap_start": int(top_run[0]),
                "gap_width": int(top_missing.size),
            }
        )

    bottom_missing = np.flatnonzero(~mask[-1, :])
    bottom_run = _run_bounds(bottom_missing)
    if (
        bottom_run is not None
        and 2 <= int(bottom_missing.size) <= 4
        and bottom_run[0] >= 1
        and bottom_run[1] <= w - 2
        and bool(mask[0, :].all())
        and bool(mask[:, 0].all())
        and bool(mask[:, -1].all())
    ):
        candidates.append(
            {
                "side": "bottom",
                "gap_start": int(bottom_run[0]),
                "gap_width": int(bottom_missing.size),
            }
        )

    left_missing = np.flatnonzero(~mask[:, 0])
    left_run = _run_bounds(left_missing)
    if (
        left_run is not None
        and 2 <= int(left_missing.size) <= 4
        and left_run[0] >= 1
        and left_run[1] <= h - 2
        and bool(mask[0, :].all())
        and bool(mask[-1, :].all())
        and bool(mask[:, -1].all())
    ):
        candidates.append(
            {
                "side": "left",
                "gap_start": int(left_run[0]),
                "gap_width": int(left_missing.size),
            }
        )

    right_missing = np.flatnonzero(~mask[:, -1])
    right_run = _run_bounds(right_missing)
    if (
        right_run is not None
        and 2 <= int(right_missing.size) <= 4
        and right_run[0] >= 1
        and right_run[1] <= h - 2
        and bool(mask[0, :].all())
        and bool(mask[-1, :].all())
        and bool(mask[:, 0].all())
    ):
        candidates.append(
            {
                "side": "right",
                "gap_start": int(right_run[0]),
                "gap_width": int(right_missing.size),
            }
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    return bool(mask.all())


def _cargo_score(container: dict, cargo: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = container["bbox"]
    cr0, cc0, cr1, cc1 = cargo["bbox"]
    side = container["side"]
    large = 10**6
    if side == "top":
        outside = cr1 < r0
        distance = r0 - cr1 if outside else large
        offset = abs(cc0 - (c0 + container["gap_start"]))
    elif side == "bottom":
        outside = cr0 > r1
        distance = cr0 - r1 if outside else large
        offset = abs(cc0 - (c0 + container["gap_start"]))
    elif side == "left":
        outside = cc1 < c0
        distance = c0 - cc1 if outside else large
        offset = abs(cr0 - (r0 + container["gap_start"]))
    else:
        outside = cc0 > c1
        distance = cc0 - c1 if outside else large
        offset = abs(cr0 - (r0 + container["gap_start"]))
    return (0 if outside else 1, int(distance), int(offset), -int(cargo["area"]))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    containers: list[dict] = []
    cargos_by_color: dict[int, list[dict]] = {}

    for comp in _components(g, bg):
        mask = _bbox_mask(comp)
        container = _detect_container(mask)
        if container is not None:
            r0, c0, r1, c1 = comp["bbox"]
            containers.append(
                {
                    "color": int(comp["color"]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "side": container["side"],
                    "gap_start": int(container["gap_start"]),
                    "gap_width": int(container["gap_width"]),
                }
            )
            continue

        if _is_solid_rectangle(mask):
            r0, c0, r1, c1 = comp["bbox"]
            cargo = {
                "color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
            cargos_by_color.setdefault(int(comp["color"]), []).append(cargo)

    used: set[tuple[int, int, int, int, int]] = set()
    for container in containers:
        matches = []
        for cargo in cargos_by_color.get(container["color"], []):
            key = (
                int(cargo["color"]),
                int(cargo["bbox"][0]),
                int(cargo["bbox"][1]),
                int(cargo["bbox"][2]),
                int(cargo["bbox"][3]),
            )
            if key in used:
                continue
            if container["side"] in ("top", "bottom"):
                if cargo["w"] != container["gap_width"]:
                    continue
            else:
                if cargo["h"] != container["gap_width"]:
                    continue
            matches.append(cargo)

        if not matches:
            continue
        cargo = min(matches, key=lambda item: _cargo_score(container, item))
        used.add(
            (
                int(cargo["color"]),
                int(cargo["bbox"][0]),
                int(cargo["bbox"][1]),
                int(cargo["bbox"][2]),
                int(cargo["bbox"][3]),
            )
        )

        cr0, cc0, cr1, cc1 = cargo["bbox"]
        out[cr0 : cr1 + 1, cc0 : cc1 + 1] = bg

        r0, c0, r1, c1 = container["bbox"]
        gap_start = int(container["gap_start"])
        if container["side"] == "top":
            nr0 = r1 - cargo["h"]
            nc0 = c0 + gap_start
        elif container["side"] == "bottom":
            nr0 = r0 + 1
            nc0 = c0 + gap_start
        elif container["side"] == "left":
            nr0 = r0 + gap_start
            nc0 = c1 - cargo["w"]
        else:
            nr0 = r0 + gap_start
            nc0 = c0 + 1

        out[nr0 : nr0 + cargo["h"], nc0 : nc0 + cargo["w"]] = int(container["color"])

    return out


BG = 0
SIDES = ("top", "bottom", "left", "right")
MIN_PAIRS = 2
MAX_PAIRS = 4
GRID_MIN = 20
GRID_MAX = 32
PLACE_PADDING = 1
PLACE_BORDER = 1
MAX_ATTEMPTS = 400


def _paint_container(
    patch: np.ndarray,
    color: int,
    top: int,
    left: int,
    h: int,
    w: int,
    side: str,
    gap_start: int,
    gap_width: int,
) -> None:
    patch[top, left : left + w] = color
    patch[top + h - 1, left : left + w] = color
    patch[top : top + h, left] = color
    patch[top : top + h, left + w - 1] = color
    if side == "top":
        patch[top, left + gap_start : left + gap_start + gap_width] = BG
    elif side == "bottom":
        patch[top + h - 1, left + gap_start : left + gap_start + gap_width] = BG
    elif side == "left":
        patch[top + gap_start : top + gap_start + gap_width, left] = BG
    else:
        patch[top + gap_start : top + gap_start + gap_width, left + w - 1] = BG


def _sample_distinct_offset(rng: np.random.Generator, limit: int, target: int) -> int:
    options = [x for x in range(limit + 1) if x != target]
    return int(rng.choice(options))


def _make_pair(rng: np.random.Generator, color: int, side: str) -> tuple[np.ndarray, np.ndarray]:
    gap_width = int(rng.integers(2, 5))
    dist = int(rng.integers(1, 4))

    if side in ("top", "bottom"):
        w_low = max(gap_width + 4, 7)
        w = int(rng.integers(w_low, 13))
        h = int(rng.integers(6, 11))
        cargo_h = int(rng.integers(1, min(4, h - 2)))
        cargo_w = gap_width
        gap_start = int(rng.integers(1, w - gap_width))
        cargo_c0 = _sample_distinct_offset(rng, w - cargo_w, gap_start)
        patch_h = h + dist + cargo_h
        patch_w = w
        inp = np.full((patch_h, patch_w), BG, dtype=int)
        out = inp.copy()

        if side == "top":
            cont_r0 = cargo_h + dist
            cargo_r0 = 0
            final_r0 = cont_r0 + h - cargo_h - 1
        else:
            cont_r0 = 0
            cargo_r0 = h + dist
            final_r0 = 1
        cont_c0 = 0

        _paint_container(inp, color, cont_r0, cont_c0, h, w, side, gap_start, gap_width)
        _paint_container(out, color, cont_r0, cont_c0, h, w, side, gap_start, gap_width)
        inp[cargo_r0 : cargo_r0 + cargo_h, cargo_c0 : cargo_c0 + cargo_w] = color
        out[final_r0 : final_r0 + cargo_h, gap_start : gap_start + cargo_w] = color
        return inp, out

    h_low = max(gap_width + 4, 7)
    h = int(rng.integers(h_low, 13))
    w = int(rng.integers(6, 11))
    cargo_h = gap_width
    cargo_w = int(rng.integers(1, min(4, w - 2)))
    gap_start = int(rng.integers(1, h - gap_width))
    cargo_r0 = _sample_distinct_offset(rng, h - cargo_h, gap_start)
    patch_h = h
    patch_w = w + dist + cargo_w
    inp = np.full((patch_h, patch_w), BG, dtype=int)
    out = inp.copy()

    if side == "left":
        cont_c0 = cargo_w + dist
        cargo_c0 = 0
        final_c0 = cont_c0 + w - cargo_w - 1
    else:
        cont_c0 = 0
        cargo_c0 = w + dist
        final_c0 = 1
    cont_r0 = 0

    _paint_container(inp, color, cont_r0, cont_c0, h, w, side, gap_start, gap_width)
    _paint_container(out, color, cont_r0, cont_c0, h, w, side, gap_start, gap_width)
    inp[cargo_r0 : cargo_r0 + cargo_h, cargo_c0 : cargo_c0 + cargo_w] = color
    out[gap_start : gap_start + cargo_h, final_c0 : final_c0 + cargo_w] = color
    return inp, out


def _place_rectangles(
    rng: np.random.Generator, shapes: list[tuple[int, int]], h: int, w: int
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda idx: -(shapes[idx][0] * shapes[idx][1]))
    for idx in order:
        ph, pw = shapes[idx]
        placed = False
        for _ in range(200):
            r0 = int(rng.integers(PLACE_BORDER, h - ph - PLACE_BORDER + 1))
            c0 = int(rng.integers(PLACE_BORDER, w - pw - PLACE_BORDER + 1))
            rr0 = max(0, r0 - PLACE_PADDING)
            rr1 = min(h, r0 + ph + PLACE_PADDING)
            cc0 = max(0, c0 - PLACE_PADDING)
            cc1 = min(w, c0 + pw + PLACE_PADDING)
            if bool(occupied[rr0:rr1, cc0:cc1].any()):
                continue
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            positions[idx] = (int(r0), int(c0))
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in positions if pos is not None]


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_pairs = int(rng.integers(MIN_PAIRS, MAX_PAIRS + 1))
    colors = [int(x) for x in rng.choice(np.arange(1, 10), size=n_pairs, replace=False)]
    sides = [SIDES[int(x)] for x in rng.integers(0, len(SIDES), size=n_pairs)]

    in_patches: list[np.ndarray] = []
    out_patches: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    for color, side in zip(colors, sides):
        inp_patch, out_patch = _make_pair(rng, color, side)
        in_patches.append(inp_patch)
        out_patches.append(out_patch)
        shapes.append(tuple(map(int, inp_patch.shape)))

    heights = sorted((shape[0] for shape in shapes), reverse=True)
    widths = sorted((shape[1] for shape in shapes), reverse=True)
    need_h = heights[0] + (heights[1] if len(heights) >= 3 else 0) + 4
    need_w = widths[0] + (widths[1] if len(widths) >= 2 else 0) + 4
    grid_h_low = max(GRID_MIN, need_h)
    grid_w_low = max(GRID_MIN, need_w)
    if grid_h_low > GRID_MAX or grid_w_low > GRID_MAX:
        return None

    grid_h = int(rng.integers(grid_h_low, GRID_MAX + 1))
    grid_w = int(rng.integers(grid_w_low, GRID_MAX + 1))
    positions = _place_rectangles(rng, shapes, grid_h, grid_w)
    if positions is None:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    for patch, pos in zip(in_patches, positions):
        _paste(inp, patch, pos[0], pos[1])
    for patch, pos in zip(out_patches, positions):
        _paste(out, patch, pos[0], pos[1])

    if np.array_equal(inp, out):
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
