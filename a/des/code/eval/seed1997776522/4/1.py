from __future__ import annotations

import random

import numpy as np


SYMMETRY_TYPES = ("horizontal_only", "vertical_only", "both")
MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_RANGE = (20, 31)
INNER_SIZE_CHOICES = (3, 4)
FRAME_COUNT = 3
BLOCK_SIZE = 2
PADDING = 1
COLOR_POOL = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _full_patch(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _border_is_uniform(patch: np.ndarray) -> bool:
    border_color = int(patch[0, 0])
    return bool(
        np.all(patch[0, :] == border_color)
        and np.all(patch[-1, :] == border_color)
        and np.all(patch[:, 0] == border_color)
        and np.all(patch[:, -1] == border_color)
    )


def _symmetry_type(pattern: np.ndarray) -> str:
    horizontal = np.array_equal(pattern, np.flipud(pattern))
    vertical = np.array_equal(pattern, np.fliplr(pattern))
    if horizontal and vertical:
        return "both"
    if horizontal:
        return "horizontal_only"
    if vertical:
        return "vertical_only"
    raise AssertionError("inner pattern must match one of the allowed symmetry classes")


def _transform_frame_patch(patch: np.ndarray, fill_color: int, symmetry: str) -> np.ndarray:
    out = np.array(patch, copy=True)
    out[0, 0] = fill_color
    out[0, -1] = fill_color
    out[-1, 0] = fill_color
    out[-1, -1] = fill_color
    if symmetry == "horizontal_only":
        return np.fliplr(out)
    if symmetry == "vertical_only":
        return np.flipud(out)
    if symmetry == "both":
        return out
    raise AssertionError("unknown symmetry type")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    comps = _find_components(g, bg)

    frames: list[dict[str, object]] = []
    blocks: list[dict[str, int]] = []

    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(comp["area"])  # type: ignore[index]
        patch = _full_patch(g, (r0, c0, r1, c1))

        if h == BLOCK_SIZE and w == BLOCK_SIZE and area == BLOCK_SIZE * BLOCK_SIZE and np.all(patch == patch[0, 0]):
            blocks.append({"r0": int(r0), "c0": int(c0), "color": int(patch[0, 0])})
            continue

        if h == w and h in (5, 6) and area == h * w and _border_is_uniform(patch):
            inner = patch[1:-1, 1:-1]
            frames.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "symmetry": _symmetry_type(inner),
                    "patch": patch,
                }
            )

    assert len(blocks) == FRAME_COUNT, "expected exactly three control blocks"
    assert len(frames) == FRAME_COUNT, "expected exactly three square frames"

    blocks.sort(key=lambda block: (int(block["c0"]), int(block["r0"])))
    color_by_symmetry = {
        "horizontal_only": int(blocks[0]["color"]),
        "vertical_only": int(blocks[1]["color"]),
        "both": int(blocks[2]["color"]),
    }

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        symmetry = str(frame["symmetry"])
        patch = np.array(frame["patch"], copy=True)  # type: ignore[arg-type]
        transformed = _transform_frame_patch(patch, color_by_symmetry[symmetry], symmetry)
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed
    return out


def _orbit_transforms(size: int, symmetry: str, cell: tuple[int, int]) -> set[tuple[int, int]]:
    orbit = {cell}
    frontier = [cell]
    while frontier:
        r, c = frontier.pop()
        next_cells: list[tuple[int, int]] = []
        if symmetry in {"horizontal_only", "both"}:
            next_cells.append((size - 1 - r, c))
        if symmetry in {"vertical_only", "both"}:
            next_cells.append((r, size - 1 - c))
        for nxt in next_cells:
            if nxt not in orbit:
                orbit.add(nxt)
                frontier.append(nxt)
    return orbit


def _symmetry_orbits(size: int, symmetry: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = sorted(_orbit_transforms(size, symmetry, (r, c)))
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _make_pattern(size: int, palette: list[int], symmetry: str) -> np.ndarray:
    orbits = _symmetry_orbits(size, symmetry)
    for _ in range(200):
        pattern = np.empty((size, size), dtype=int)
        chosen = [random.choice(palette) for _ in orbits]
        if len(set(chosen)) < 2:
            continue
        for orbit, color in zip(orbits, chosen):
            for r, c in orbit:
                pattern[r, c] = int(color)
        if _symmetry_type(pattern) == symmetry:
            return pattern
    raise RuntimeError(f"failed to create a {symmetry} pattern")


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = PADDING) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _rect_candidates(
    occupied: np.ndarray,
    height: int,
    width: int,
    forbidden_lefts: set[int] | None = None,
) -> list[tuple[int, int]]:
    forbidden_lefts = forbidden_lefts or set()
    h, w = occupied.shape
    candidates: list[tuple[int, int]] = []
    for top in range(h - height + 1):
        for left in range(w - width + 1):
            if left in forbidden_lefts:
                continue
            if _can_place(occupied, top, left, height, width):
                candidates.append((top, left))
    return candidates


def _occupy(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _paint_frame(grid: np.ndarray, top: int, left: int, border_color: int, pattern: np.ndarray) -> None:
    size = int(pattern.shape[0] + 2)
    grid[top : top + size, left : left + size] = int(border_color)
    grid[top + 1 : top + size - 1, left + 1 : left + size - 1] = pattern


def _build_expected(
    grid: np.ndarray,
    frames: list[dict[str, object]],
    block_colors: dict[str, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frame in frames:
        top = int(frame["top"])
        left = int(frame["left"])
        patch = np.array(frame["patch"], copy=True)  # type: ignore[arg-type]
        symmetry = str(frame["symmetry"])
        size = int(patch.shape[0])
        out[top : top + size, left : left + size] = _transform_frame_patch(
            patch, int(block_colors[symmetry]), symmetry
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    inner_size = random.choice(INNER_SIZE_CHOICES)
    frame_size = inner_size + 2
    min_side = max(GRID_SIDE_RANGE[0], frame_size + 8)
    height = random.randint(min_side, GRID_SIDE_RANGE[1] - 1)
    width = random.randint(min_side, GRID_SIDE_RANGE[1] - 1)

    palette = list(COLOR_POOL)
    random.shuffle(palette)
    bg = palette[0]
    frame_colors = palette[1:4]
    block_colors = {
        "horizontal_only": palette[4],
        "vertical_only": palette[5],
        "both": palette[6],
    }
    pattern_palette = palette[7:]
    if len(pattern_palette) < 2:
        return None

    patterns = {
        "horizontal_only": _make_pattern(inner_size, pattern_palette, "horizontal_only"),
        "vertical_only": _make_pattern(inner_size, pattern_palette, "vertical_only"),
        "both": _make_pattern(inner_size, pattern_palette, "both"),
    }

    occupied = np.zeros((height, width), dtype=bool)
    frames: list[dict[str, object]] = []
    frame_order = list(SYMMETRY_TYPES)
    random.shuffle(frame_order)

    for border_color, symmetry in zip(frame_colors, frame_order):
        candidates = _rect_candidates(occupied, frame_size, frame_size)
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _occupy(occupied, top, left, frame_size, frame_size)
        patch = np.full((frame_size, frame_size), int(border_color), dtype=int)
        patch[1:-1, 1:-1] = patterns[symmetry]
        frames.append(
            {
                "top": top,
                "left": left,
                "symmetry": symmetry,
                "patch": patch,
            }
        )

    block_positions: list[tuple[int, int]] = []
    used_lefts: set[int] = set()
    for _ in range(FRAME_COUNT):
        candidates = _rect_candidates(occupied, BLOCK_SIZE, BLOCK_SIZE, forbidden_lefts=used_lefts)
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _occupy(occupied, top, left, BLOCK_SIZE, BLOCK_SIZE)
        used_lefts.add(left)
        block_positions.append((top, left))

    block_positions.sort(key=lambda rc: (rc[1], rc[0]))
    ordered_block_types = ("horizontal_only", "vertical_only", "both")

    grid = np.full((height, width), int(bg), dtype=int)
    for frame in frames:
        _paint_frame(grid, int(frame["top"]), int(frame["left"]), int(frame["patch"][0, 0]), frame["patch"][1:-1, 1:-1])  # type: ignore[index]
    for (top, left), symmetry in zip(block_positions, ordered_block_types):
        grid[top : top + BLOCK_SIZE, left : left + BLOCK_SIZE] = int(block_colors[symmetry])

    if _most_common_color(grid) != bg:
        return None

    expected = _build_expected(grid, frames, block_colors)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
